#%% import packages
import copy
import torch
from TrenoToPheno.make_treno import make_treno


#%% important classes
# Parameters of the whole process
class PARAMS:
    def __init__(self, n_alleles, n_samples):
        self.n_alleles = n_alleles
        self.n_samples = n_samples
        self.map_Gtog = None
        self.n_features = None


# Genotype
class GENO:
    def __init__(self, G=0):
        self.n_alleles = 0
        self.n_samples = 0
        self.G = G

    def parse_paras(self, paras):
        self.n_samples = paras.n_samples
        self.n_alleles = paras.N_alleles


# Translated genos
class TRENO:
    def __init__(self):
        self.n_features = 0
        self.n_samples = 0

    def parse_paras(self, paras):
        self.n_samples = paras.n_samples
        self.n_features = paras.n_features


#%% map from Geno to Treno
class MAP_GT(object):
    N_alleles = 0

    N_eQTL = None
    index_eQTL = []

    N_dominant = None
    N_gene = None
    W_eQTL = torch.tensor([])

    N_polypeptide = None

    N_protein = None
    W_poly = torch.tensor([])

    N_treno = None
    W_fold = torch.tensor([])

    _, Ltreno, Utreno, dtreno = make_treno()
    Utreno = torch.tensor(Utreno)
    Utreno_f = Utreno.flatten().to('cuda')
    Ltreno = torch.tensor(Ltreno)
    Ltreno_f = Ltreno.flatten().to('cuda')

    def __init__(self, N_alleles=0, N_treno=0):
        if N_alleles:
            self.N_alleles = N_alleles
        else:
            self.N_alleles = 1000

        if N_treno:
            self.N_treno = N_treno

    def start(self, N_eQTL=0, N_dominant=0, N_gene=0, N_protein=0, N_Treno=0):
        self.make_eQTL(N_eQTL=N_eQTL, N_dominant=N_dominant)
        self.make_gene(N_gene=N_gene)
        self.make_Polypeptide()
        self.make_Protein(N_protein=N_protein)
        self.make_Treno(N_Treno=N_Treno)

    def make_eQTL(self, N_eQTL=0, N_dominant=0):
        if N_eQTL:
            self.N_eQTL = N_eQTL
        else:
            if self.N_eQTL is None:
                self.N_eQTL = int(0.6 * self.N_alleles)
        if N_dominant:
            self.N_dominant = N_dominant
        else:
            if self.N_dominant is None:
                self.N_dominant = 0.3*self.N_eQTL

        index_eQTL = torch.randperm(self.N_alleles)[:self.N_eQTL]
        self.index_eQTL = index_eQTL

        index_dominant = torch.randperm(self.N_eQTL)[:int(self.N_dominant)]
        self.index_dominant = index_dominant

    def Geno_to_eQTL(self, Geno):
        if Geno.dim() == 2:
            additive = Geno[:, self.index_eQTL]
        else:
            eQTL = Geno[:, self.index_eQTL, :]
            additive = eQTL.sum(dim=2)
        dominant = (additive == 1) * 1.0

        eQTL = torch.cat((additive, dominant), -1)
        index = torch.cat((torch.arange(0, self.N_eQTL), self.index_dominant+self.N_eQTL), 0)
        eQTL = eQTL[:, index.int()]
        return eQTL

    def make_gene(self, N_gene=0):
        if N_gene:
            self.N_gene = N_gene
        else:
            if self.N_gene is None:
                self.N_gene = 2*self.N_eQTL
        self.W_eQTL = torch.randn(int(self.N_eQTL+self.N_dominant+1), self.N_gene)

    def eQTL_to_Gene(self, eQTL):
        X = torch.cat((torch.ones(len(eQTL), 1), eQTL), 1)
        Gene = torch.matmul(X, self.W_eQTL)
        return Gene

    def make_Polypeptide(self):
        self.N_polypeptide = self.N_gene

    def Gene_to_Polypeptide(self, Gene):
        Polypeptide = torch.relu(Gene)
        self.N_polypeptide = Polypeptide.shape[-1]
        return Polypeptide

    def make_Protein(self, N_protein=0):
        if N_protein:
            self.N_protein = N_protein
        else:
            if self.N_protein is None:
                self.N_protein = 2*self.N_polypeptide
        self.W_poly = torch.randn(int(self.N_polypeptide + 1), int(self.N_protein))

    def Polypeptide_to_Protein(self, Polypeptide):
        X = torch.cat((torch.ones(len(Polypeptide), 1), Polypeptide), 1)
        Protein = torch.matmul(X, self.W_poly)
        return Protein

    def make_Treno(self, N_Treno=0):
        if N_Treno:
            self.N_treno = N_Treno
        else:
            if self.N_treno is None:
                self.N_treno = 51 * 3
        self.W_fold = torch.randn(int(self.N_protein + 1), int(self.N_treno))

    def Protein_to_Treno(self, Protein):
        X = torch.cat((torch.ones(len(Protein), 1), Protein), 1)
        Treno = torch.matmul(X, self.W_fold)
        Treno = torch.sigmoid(Treno)
        return Treno

    def get_Treno(self, Geno):
        eQTL = self.Geno_to_eQTL(Geno)
        GS = self.eQTL_to_Gene(eQTL)
        PS = self.Gene_to_Polypeptide(GS)
        PrS = self.Polypeptide_to_Protein(PS)
        Treno = self.Protein_to_Treno(PrS)

        # return Treno
        N_sample = len(Treno)
        Treno_t = Treno.reshape([N_sample, 3, 51])
        Treno_t = Treno_t * (self.Utreno - self.Ltreno) + self.Ltreno
        Treno_t = torch.maximum(Treno_t, self.Ltreno)
        Treno_t = torch.minimum(Treno_t, self.Utreno)
        Treno_t = Treno_t.detach()

        return Treno_t


#%% main
if __name__ == '__main__':
    m = MAP_GT()
    Geno = torch.randint(2, (1000, m.N_alleles, 2))
    Geno = Geno.float().requires_grad_()

    m.make_eQTL()
    eQTL = m.Geno_to_eQTL(Geno)

    m.make_gene()
    Gene = m.eQTL_to_Gene(eQTL)

    m.make_Polypeptide()
    Poly = m.Gene_to_Polypeptide(Gene)

    m.make_Protein()
    Protein = m.Polypeptide_to_Protein(Poly)

    m.make_Treno()
    Treno = m.Protein_to_Treno(Protein)

    dist = (Treno - 0.5).abs().sum()
    W_fold_temp = copy.deepcopy(m.W_fold)
    dim0 = W_fold_temp.shape[0]
    dim1 = W_fold_temp.shape[1]
    break_flag = 0

    while dist > 10:
        break_flag = 0
        for i in range(dim0):
            for j in range(dim1):
                W_fold_temp[i][j] += 1
                Treno_temp = torch.matmul(Protein, W_fold_temp)
                Treno_temp = torch.sigmoid(Treno_temp)
                dist_temp = (Treno_temp - 0.5).abs().sum()

                if dist_temp < dist:
                    print('Old distance:', float(dist), 'New distance:', float(dist_temp))

                    m.W_fold = W_fold_temp
                    dist = dist_temp
                    break_flag = 1
                else:
                    W_fold_temp = copy.deepcopy(m.W_fold)

                if break_flag:
                    break

                W_fold_temp[i][j] -= 1
                Treno_temp = torch.matmul(Protein, W_fold_temp)
                Treno_temp = torch.sigmoid(Treno_temp)
                dist_temp = (Treno_temp - 0.5).abs().sum()

                if dist_temp < dist:
                    print('Old distance:', float(dist), 'New distance:', float(dist_temp))

                    m.W_fold = W_fold_temp
                    dist = dist_temp
                    break_flag = 1
                else:
                    W_fold_temp = copy.deepcopy(m.W_fold)

                if break_flag:
                    break

            if break_flag:
                break


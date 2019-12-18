#!/usr/bin/env python
# coding: utf-8

# # $f(K)$ metrikos naudojimo pavyzdys K-vidurkių klasterizavimo rezultatams

# In[10]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[11]:


from IPython.display import Image
import plotnine as gg
import pandas as pd


# `pyspark.ml.clustering.KMeansModel` metodas `computeCost` apskaičiuoja stebėjimų Euklido atstumų nuo savo klasterių  centrų sumą $S_K$ (angl. _Within Set Sum of Squared Error (WSSSE)_):
# 
# $I_k = \sum_{\mathbf{x}_i \in C_k} \| \mathbf{x}_i - \mathbf{\overline{x}}_k \|$
# 
# $S_K = \sum_{k}^{K} I_k$
# 
# čia 
# 
# $k$ - klasterio indeksas,
# 
# $C_k$ - $k$-asis klasteris
# 
# $K$ - klasterių skaičius,
# 
# $N_k$ - $k$-jam klasteriui priklausančių stebėjimų skaičius,
# 
# $\mathbf{x_i}$ - $i$-tojo stebėjimo vektorius,
# 
# $\mathbf{\overline{x}}_k$ - $k$-otojo klasterio vidurinio taško (centro) vektorius,
# 
# $\|\mathbf{x}\|$ - vektoriaus Euklido norma, t.y. kvadratinė šaknis iš jo komponenčių kvadratų sumos.

# $f(K)$ yra naudojama nustatyti optimalią $K$ reikšmę ir yra aprašyta [čia](http://www.ee.columbia.edu/~dpwe/papers/PhamDN05-kmeans.pdf) ir [čia](https://datasciencelab.wordpress.com/2014/01/21/selection-of-k-in-k-means-clustering-reloaded/). Jos reikšmė $f(K)$ apskaičiuojama tokiu būdu:

# In[12]:


Image("../data/img/fk.png")


# Optimilaus $K$ yra ties mažiausia $f(K)$ reikšme.

# Realizuojame formulės išraišką.

# In[13]:


def compute_fk(k, sse, prev_sse, dim):
    if k == 1 or prev_sse == 0:
        return 1
    weight = weight_factor(k, dim)
    return sse / (weight * prev_sse)

# calculating alpha_k in functional style with tail recursion -- which is not optimized in Python :(
def weight_factor(k, dim):
    if not k > 1:
        raise ValueError("k must be greater than 1")
        
    def weigth_factor_accumulator(acc, k):
        if k == 2:
            return acc
        return weigth_factor_accumulator(acc + (1 - acc) / 6, k - 1)
        
    weight_k2 = 1 - 3 / (4 * dim)
    return weigth_factor_accumulator(weight_k2, k)


# Aprašome funkciją, kuri iš $K$ ir $S_K$ reikšmių porų `list`'o pateikia galimas įvertinti $f(K)$ reikšmes.

# In[14]:


def compute_fk_from_k_sse_pairs(k_sse_pairs, dimension):
    triples = make_fk_triples(k_sse_pairs)
    k_fk_pairs = [
        (k, compute_fk(k, sse, prev_sse, dimension))
        for (k, sse, prev_sse) in triples]
    return sorted(k_fk_pairs, key=lambda pair: pair[0])


def make_fk_triples(k_sse_pairs):
    sorted_pairs = sorted(k_sse_pairs, reverse=True)
    candidates = list(zip(sorted_pairs, sorted_pairs[1:] + [(0, 0.0)]))
    triples = [
        (k, sse, prev_sse)
        for ((k, sse), (prev_k, prev_sse)) in candidates
        if k - prev_k == 1
    ]
    return triples


# Naudojimo pavyzdys:

# In[15]:


get_ipython().system(' cat ../data/examples_io/metrics__k_means__sse.jsonl')


# Tarkime, iš disko nuskaitome tokią $K$ ir $S_K$ reikšmių lentelę.

# In[16]:


metrics_pddf = pd.read_json(
    "../data/examples_io/metrics__k_means__sse.jsonl", 
    orient="records",
    lines=True)

metrics_pddf


# Pakeičiame stulpelių tvarką.

# In[17]:


k_sse_pddf = metrics_pddf[["k", "sse"]]

k_sse_pddf


# Skačiuojant $f(K)$ metriką reikia žinoti duomenų dimensiją, t.y. klasterizavimui naudotų požymių skaičių. Tarkime, kad šiuo atveju naudojome du požymius.

# In[18]:


dimension = 2 

k_sse_pairs = [tuple(r) for r in k_sse_pddf.to_records(index=False)]
k_sse_pairs


# In[19]:


k_fk_pairs = compute_fk_from_k_sse_pairs(k_sse_pairs, dimension)
k_fk_pairs


# In[20]:


k_fk_pddf = pd.DataFrame.from_records(k_fk_pairs, columns=["k", "fk"])
k_fk_pddf


# In[21]:


plot_k_sse = (
    gg.ggplot(gg.aes(x="k", y="sse"), data=k_sse_pddf) + 
    gg.geom_line() + 
    gg.xlab("K") +
    gg.ylab("SSE") + 
    gg.ggtitle("SSE pagal klasterių skaičių K") +
    gg.theme_bw()
)

plot_k_fk = (
    gg.ggplot(gg.aes(x="k", y="fk"), data=k_fk_pddf) + 
    gg.geom_line() + 
    gg.xlab("K") +
    gg.ylab("f(K)") + 
    gg.ggtitle("f(K) pagal klasterių skaičių K") +
    gg.theme_bw()
)


# In[22]:


print(plot_k_sse)
print(plot_k_fk)


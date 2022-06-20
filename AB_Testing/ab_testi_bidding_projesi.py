
############################

############################

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# pip install statsmodels
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)                  # MERVE İMRENNNN
pd.set_option('display.expand_frame_repr', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

dataframe_control = pd.read_excel("datasets/ab_testing.xlsx",sheet_name="Control Group")
dataframe_test = pd.read_excel("datasets/ab_testing.xlsx", sheet_name="Test Group")

df_control = dataframe_control.copy()
df_test = dataframe_test.copy()

# Impression: REklam görüntüleme sayısı
# Click: Görüntülenen reklama tıklama sayısı
# Purchase: Tıklanan reklamlar sonrası satın almınan ürün sayısı
# Earning: SAtın alınan ürünler sonrası elde edilen kazanç

df_test.head()
df_control.head()

# Kontrol ve test grupları birleştirme
df_control["group"] = "control"
df_test["group"] = "test"

df = pd.concat([df_control, df_test], axis=0, ignore_index=False) # alt alta birleştirdik
df.head()

# Purchase ortalamasını group lara göre incele
df.groupby("group").agg({"Purchase": "mean"})

###########################
# AB TEstinin Hipotezinin TAnımlanması
###########################

# 1. Hipotezi tanımlama
# H0: M1 = M2 Kontrol grubu ve test grubu satın alma ortalamarı arsında fark yoktur.
# H1: M1 != M2 Kontrol grubu ve test grubu satın alma ortalamaları arasında fark vardır.
# p < 0.05  ise H0 reddedilir.

# 2. varsayımları inceleme
# 2.1 normallik varsayımı
# H0: Normal dağılım varsayımı sağlamaktadır.

df.groupby("group").agg({"Purchase": "mean"})

test_stat, pvalue = shapiro(df.loc[df["group"] == "control", "Purchase"])
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))
# pvalue = 0.58 H0 reddedilemez
# Control grubunun değerleri normal dağılım varsayımını sağlar
test_stat, pvalue = shapiro(df.loc[df["group"] == "test", "Purchase"])
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))
# pvalue = 0.15 H0 reddedilemez
# Normal dağılım varsayımı sağlamaktadir

# *** Normal dağılım varsayımı sağlanmasaydı varyans homojenliğine bakmadan nonparametrik t testine geçilecektir.

# 2.2 Varyans homojenliği
# H0: Varyanslar homojendir.
# H1: VAryanslar homojen değildir.

test_stat, pvalue = levene(df.loc[df["group"] == "control", "Purchase"],
                           df.loc[df["group"] == "test", "Purchase"])

print("Test Stat = %4.f, p-value = %.4f" % (test_stat, pvalue))
# p-value = 0.10 çıktı H0 Reddedilemez
# Varyans homojendir.

# 3. Hipotezin Uygulanması
# 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi(parametrik test)
# 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi (non- parametrik test)

# Varsayımlar sağlanmaktadır bu yüzden t testi parametrik test yapılacaktır.

test_stat, pvalue = ttest_ind(df.loc[df["group"] == "control", "Purchase"],
                              df.loc[df["group"] == "test", "Purchase"],
                              equal_var=True) # varyans homojen olduğu için True homojen olmasaydı False oalcaktı

print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))
# p-value = 034 H0 reddedilemez

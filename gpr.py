import numpy as np
import matplotlib.pyplot as plt

# Gauss Süreci çekirdek fonksiyonu
# Hiperparametreler: 
#                   lamda: uzunluk ölçeği
#                   var_f: benzerlik fonksiyonunun varyans parametresi
def kernel(a, b, lamda, var_f):
        return var_f*np.exp(-.5 * (1/lamda)**2 * (np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)))

# öncül dağılımdan alınacak örnek sayısı - çok değişkenli Gauss dağılımının boyut sayısı 
N = 50

# tahmin edilecek fonksiyon sayısı
Nstar = 3

# çekirdek fonksiyonu hesabında varyansı temsil eden hiper-parametre 
# (şimdilik basit olsun diye varyansın 1 olduğunu varsayalım)
var_f = (1)**2

# fonksiyonun uzunluk ölçeğini temsil eden hiper-parametre
l = 0.9

# y = f(x) + err ifadesindeki  gözlem hatasının (gürültü) olasılık dağılımının standart sapması
stdv_y = 0    

# tahmin edeceğimiz fonksiyonların beklenen değerleri (ortalaması)
mu = 0

# fonksiyon değerlerini tahmin edeceğimiz noktaların x eksenindeki indeksleri (-10 ile 10 arasında)
X_test = np.linspace(-10, 10, N).reshape(-1,1)

# test noktalarının birbirine benzerliğini hesapla (var_f bir hiper paratmetre)
Kss = kernel(X_test, X_test, l, var_f)

# numerik kararlılığı sağlayacak kadar küçük bir sayı (K'nın eigen-değerleri hızla küçülebilir) seç
eps = 1e-10

# Cholesky ayrıştımasını yap ve kovaryansın karekökü L 'yi geri döndür
L_ss = mu + np.linalg.cholesky(Kss + eps*np.eye(N))

# standart normal ve L'yi kullanarak öncül dağılımı bul: L*N~(0, I) 
fprior = L_ss @ np.random.normal(size=(N, Nstar))

# öncül dağılımdan örnek olarak çekilen fonksiyonları çizdir
plt.plot(X_test, fprior)
plt.axis([-10, 10, -3, 3])
plt.title('Gauss Süreci öncül dağılımından örneklenmiş %i fonksiyon' % Nstar)
plt.show()

# kovaryans fonksiyonunu çizdir
plt.title("Öncül kovaryans $K(X_*, X_*)$")
plt.contour(Kss)
plt.show()

# fonksiyon gözlemleri ile eşleşen X indekslerini oluştur (eğitim verisi sayılır) 
X_train = np.array([-4, -3, -2, -1, 1, 2, 3, 4, 5, 6]).reshape(10,1) 

# stdv_y'yi sıfır seçtiğimizde gürültüsüz (gözlem hatası olmayan) durumu (y= f(x) + 0) simüle ediyoruz
y_train = np.sin(X_train) + stdv_y*np.random.randn(10,1) 

# gözlemlerle eşleşen X indeksleriyle çekirdek fonksiyonunu hesaplayalım 
K = kernel(X_train, X_train, l, var_f)

# kovaryans matrisinin karekökünü bulalım 
L = np.linalg.cholesky(K + stdv_y*np.eye(len(X_train)))

# test noktalarında beklenen değerleri (ortalama vektörü) hesapla 
Ks = kernel(X_train, X_test, l, var_f)
Lk = np.linalg.solve(L, Ks)
mu = np.dot(Lk.T, np.linalg.solve(L, y_train)).reshape((N,))

# standart sapmayı hesapla 
s2 = np.diag(Kss) - np.sum(Lk**2, axis=0)
stdv = np.sqrt(s2)

# test noktalarında sonsal dağılımdan örnek fonksiyonlar çek 
L = np.linalg.cholesky(Kss + 1e-6*np.eye(N) - np.dot(Lk.T, Lk))
fpost = mu.reshape(-1,1) + np.dot(L, np.random.normal(size=(N,3)))

plt.plot(X_train, y_train, 'bs', ms=8, label='Gözlem')
plt.plot(X_test, fpost)
plt.gca().fill_between(X_test.flat, mu-2*stdv, mu+2*stdv, color="#dddddd", label='95% Güven Aralığı')
plt.plot(X_test, mu, 'r--', lw=2, label='Beklenen')
plt.axis([-10, 10, -3, 3])
plt.title('GS sonsal dağılımdan örneklenmiş 3 fonksiyon')
plt.legend(loc='lower left')
plt.show()


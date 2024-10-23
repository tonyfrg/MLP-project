import numpy as np
import matplotlib.pyplot as plt

### On choisit des points qui sont dans le carré [0,1]x[0,1] ###
X = np.array([0.5, 0.3, 0.7, 0.6, 0.9, 0.8, 0.2, 0.3, 0.6, 0.4])
Y = np.array([0.7, 0.1, 0.5, 0.8, 0.2, 0.9, 0.5, 0.8, 0.1, 0.4])

### On décide qui est gentil (1,0) et qui est méchant (0,1)
y_X_1 = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
y_X_2 = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 0])

### Affectation gentil/méchant ###
M_X = []
M_Y = []
G_X = []
G_Y = []
for i in range(len(y_X_1)):
    if y_X_1[i] == 0:
        G_X.append(X[i])
        G_Y.append(Y[i])
    elif y_X_1[i] == 1:
        M_X.append(X[i])
        M_Y.append(Y[i])
### Affichage point ###
plt.figure(1)
plt.scatter(G_X, G_Y, c='g', marker='o', label='les gentils')
plt.scatter(M_X, M_Y, c='b', marker='^', label='les méchants')
plt.legend(loc='best')
plt.title('Jeu de données')
plt.show()

## un peu de blabla pour y voir plus clair ##
"""
On va créer un réseau de neurone avec :
- L couches
- n_l neurones sur la l-ième couche
- condition de départ : n_1 = 2
- condition d'arrivée : n_L = 2
- on nomme z_l le vecteur d'entrée, composée de z_l_j sa valeur sur la j-ième ligne
- on nomme a_l le vecteur de sortie, obtenu par la sigmoide de z_l
- on nomme W la matrice contenant les coefficients du neurone, tel que :
    * w_j_k le coefficient de la j-ième ligne, k-ième colonne
    * les coefs de la j-ième ligne donneront les poids pour chaque a_k
    * on obtient le vecteur qui sur la ligne j donnera
            sum_{i=1}{n_l}(w_j_k * a_k) = (w_j_1 * a_1 + w_j_2 * a_2 + ... + w_j_i * a_i)
"""


### Couches de neurones ###
print('Pour rappel, la première et dernière couche sont fixées :', "Taille d'entrée : 2", "Taille de sortie : 2", sep='\n')
L = int(input('Nombre de couche : ')) ### dans l'exemple on prend L=4
n_L = [i**0 for i in range(L)]
n_L[0] = 2
n_L[-1] = 2
for i in range(1,L-1):
    print('Nombre de neurones sur la ', i + 1, '-ème couche :', end=' ')
    n_L[i] = int(input(''))
print("réseau de neurone de couche de taille respectivement :", n_L)

##### POIDS W #####
# chaque coefficient est choisi au hasard entre -1 et 1
W =[] # Liste des W_l, càd les matrices avec les poids des neurones
for i in range(1, L):
    W.append(0.5 * np.random.normal(-1, 1, (n_L[i], n_L[i-1])))

choix_affichage_W = input('Voulez-vous afficher le W généré ? [y] ')
if choix_affichage_W == 'y':
    print(W)

##### BIAIS B #####
# on met des biais aléatoires compris entre -1 et 1 #
B = []
for i in range(1,L):
    B.append(0.5 * np.random.normal(-1, 1, (n_L[i], 1)))

choix_affichage_B = input('Voulez-vous afficher le B généré ? [y] ')
if choix_affichage_B == 'y':
    print(B)

"""
Il nous faudra, pour la descente de gradient :
    - faire la passe forward, qui correspond à calculer,tout les a_l, 2 =< l =< L, donc aussi notre vecteur de sortir
    - faire la passe backward, qui permet de calculer les vecteurs delta^(l) qui nous donnera donnera l'erreur.
    - faire le décalage pour chaque colonne avec les formules que l'on a calculé càd :
        w_i,j^(l) = w_i,j^(l) + p * delta_i^(l) * a_j^(l-1)
        b_j^(l) = b_j^(l) + p * delta_j^(l)
"""
### Apport des éléments de la descente de gradient ###
p = 0.6  # p pour pas, de la descente de gradient
N = 100000  # N pour nombres, d'itérations
Loss = np.zeros([N, 1]) # Récupération des taux d'erreurs

def sigm(x): # sigmoide, permettant de passer de 'z^(l)' à 'a^(l)'
    return 1/(1+np.exp(-x))

def forward(x, y, W, B): # ici, (x,y) sont les coordonnées de départ du point
    a = []
    a_l = np.array([[x],[y]])
    a.append(a_l)
    for l in range(0,L-1):
        a_l = W[l] @ a_l + B[l]
        a_l = sigm(a_l)
        a.append(a_l)
    return a

def D(a):
    d = np.eye((len(a)))
    for j in range(len(a)):
        d[j,j] = a[j,0] * (1 - a[j,0])
    return d

def backprop(x, y, a, W): # ATTENTION : [x,y] représente y(x^i) c'est-à-dire leur 'affectation', et non des coordonnées
    Delta = [i**0 for i in range(L-1)]
    delta = np.array([[x], [y]])
    delta = a[-1] - delta
    delta = D(a[-1]) @ delta
    Delta[-1] = delta
    for l in range(2, L):
        delta = W[-l+1].T @ delta
        delta = D(a[-l]) @ delta
        Delta[-l] = delta
    return Delta

def maj(W, B, a, d):
    for l in range(0, L-1):
        W[l] += - p * d[l] @ a[l].T  # pas de décalage car W[l] = W^(l+2) et a[l] = a^(l+1)
        B[l] += - p * d[l]
    return W, B

def loss(x,y,a):
    S = 0
    for i in range(len(x)):
        S += 1/2 * ((x[i] - a[i][0,0])**2 + (y[i] - a[i][1,0])**2)
    return S/len(x)

##### DESCENTE DE GRADIENT STOCHASTIQUE #####
pause = input('Lancement de la descente de gradient !')
stock_a = []  #pour le calcul du Loss à la fin de chaque epoch
for n in range(N):
    for m in range(len(X)):
        a = forward(X[m], Y[m], W, B)
        d = backprop(y_X_1[m], y_X_2[m], a,  W)
        W, B = maj(W, B, a, d)
        stock_a.append(a[-1])
    Loss[n] = loss(y_X_1, y_X_2, stock_a)
    if n % 1000 == 0:
        print('epoch n°', n, ':', Loss[n])
        print()
    stock_a.clear()

print('Loss final :', Loss[-1], end=' ')


##### AFFICHAGE COURBE LOSS #####
plt.figure(2)
plt.gca().set_yscale('log')
it = np.arange(0, N, 1)
y = Loss[0:N:1]
plt.plot(it, y, 'b')
plt.title('Courbe Loss au cours des epochs')
plt.show()

##### AFFICHAGE FRONTIERE DECISION #####
x_aff = np.linspace(0, 1, 100)
y_aff = np.linspace(0, 1, 100)
X_AFF_1 = []
Y_AFF_1 = []
X_AFF_2 = []
Y_AFF_2 = []
for x in x_aff:
    for y in y_aff :
        if forward(x, y, W, B)[-1][0,0] > forward(x, y, W, B)[-1][1,0] :
            X_AFF_1.append(x)
            Y_AFF_1.append(y)
        elif forward(x, y, W, B)[-1][0,0] < forward(x, y, W, B)[-1][1,0] :
            X_AFF_2.append(x)
            Y_AFF_2.append(y)
plt.figure(3)
plt.scatter(X_AFF_1, Y_AFF_1, c='blue', marker='s', alpha=0.03)
plt.scatter(X_AFF_2, Y_AFF_2, c='green', marker='s', alpha=0.03)
plt.scatter(G_X, G_Y, c='g', marker='o', label='les gentils')
plt.scatter(M_X, M_Y, c='b', marker='^', label='les méchants')
plt.title('Frontière de décision et données')
plt.legend(loc='best')
plt.show()
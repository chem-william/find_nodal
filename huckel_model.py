import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(
    style="white",
    rc={
        "font.family": "Liberation Sans",
        "font.size": 40,
        "axes.linewidth": 2,
        "lines.linewidth": 3,
    },
    font_scale=2.5,
    palette=sns.color_palette("Reds"),
)


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    return rho, phi


alpha = 0
beta = -3

dim = 8
huckel = np.zeros((2*dim, 2*dim))
np.fill_diagonal(huckel, alpha)

small_huckel = np.zeros((dim, dim))
indices = np.arange(0, len(small_huckel))
small_huckel[indices, indices - 1] = beta
small_huckel[0, -1] = 0
small_huckel += small_huckel.T

huckel[0:dim, 0:dim] = small_huckel
huckel[dim:, dim:] = small_huckel

eig_vals, eig_vecs = np.linalg.eigh(huckel)

length = len(huckel)//2
c2_matrix = np.zeros((huckel.shape[0], huckel.shape[1]))
for idx in range(0, len(huckel), 2):
    p_x = np.zeros(length)
    p_y = np.zeros(length)
    SALC_a = (1/np.sqrt(2))*(eig_vecs[:, idx] + eig_vecs[:, idx + 1])
    p_x[:length] = SALC_a[length:length*2]
    p_y[:length] = SALC_a[:length]
    a = np.array((p_y, p_x))

    SALC_b = (1/np.sqrt(2))*(eig_vecs[:, idx] - eig_vecs[:, idx + 1])
    p_x[:length] = SALC_b[length:length*2]
    p_y[:length] = SALC_b[:length]
    b = np.array((p_y, p_x))

    c2_matrix[idx, :] = a.ravel()
    c2_matrix[idx + 1, :] = b.ravel()

print("Energies")
print(eig_vals/beta)
print("="*25)

x = np.zeros(dim + 1)
x[:dim] = c2_matrix[6][:dim]
y = np.zeros(dim + 1)
y[1:] = c2_matrix[6][dim:]

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.plot(x, y, np.arange(len(x)), "-o", c="k")
plt.show()


degrees = []
norm = np.linalg.norm
for idx, (i, j) in enumerate(zip(x, y)):
    if idx == 0:
        degrees.append(0)
        continue
    u, v = [x[idx - 1], y[idx - 1]], [x[idx], y[idx]]
    deg = np.rad2deg(
        np.arctan2(
            np.linalg.det([u, v]),
            np.dot(u, v),
        )
    )
    degrees.append(deg)

degrees = np.array(degrees)

print(degrees)

# np.save("./degrees_homo.npy", degrees)

plt.scatter(range(len(degrees)), np.cumsum(np.abs(degrees)), c="k")
print(np.cumsum(np.abs(degrees)))

plt.plot([0, len(np.cumsum(np.abs(degrees))) - 1], [0, np.max(np.cumsum(np.abs(degrees)))], linestyle="--", c="k")
plt.ylim(0, np.max(np.cumsum(np.abs(degrees))))
plt.show()

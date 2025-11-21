import torch

# Δημιουργία vector 0..11
x = torch.arange(12, dtype=torch.float32)
print("x:", x)

# Αλλαγή σχήματος σε 3x4
X = x.reshape(3, 4)
print("X:\n", X)

# Βάλε τα στο GPU (αν υπάρχει)
device = "cuda" if torch.cuda.is_available() else "cpu"
X_gpu = X.to(device)
print("Device of X_gpu:", X_gpu.device)

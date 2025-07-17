import numpy as np
A = np.array([[0,1],[-6,-5]])
E,V=np.linalg.eig(A)
print("Quiz 1(1) Eigenvalues:", E)

A = np.array([[0,1],[-6,5]])
E,V=np.linalg.eig(A)
print("Quiz 1(2) Eigenvalues:", E)


A = np.array([[0,1],[-1,0]])
E,V=np.linalg.eig(A)
print("(3) Eigenvalues:", E)

A = np.array([[-6,18],[-2,6]])
E,V=np.linalg.eig(A)
print("(4) Eigenvalues:", E)
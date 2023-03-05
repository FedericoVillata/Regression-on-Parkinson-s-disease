import numpy as np
import matplotlib.pyplot as plt
np.random.seed(247586)

class SolveMinProb:
    def __init__(self, y, A): #inizialization; y=array of 3 element all equal to 1; A=3x3 identity matrix
        self.matr=A #matrix A
        self.Np=y.shape[0] #number of rows
        self.Nf=A.shape[1] #Number of columns
        self.vect=y #column vector y
        self.sol=np.zeros((self.Nf), dtype=float) #column vector (w_hat) of Nf floats all equal to 0
        self.n=0
        return

    def plot_w_hat(self, title='Solution'): #method to plot
        w_hat=self.sol
        n=np.arange(self.Nf) #Nf elements starting from 0
        plt.figure() #create a new figure
        plt.plot(n, w_hat) #plot n and w using default line style and color #!!! chiedere perchè dovrebbe andarci w
        plt.xlabel('n') #label on the x-axis
        plt.ylabel('w_hat(n)') #label on the y-axis
        plt.title(title) #set the title
        plt.grid() #set the grid
        plt.show() #show figure on screen
        return

    def print_result(self, title): #method to print the result
        print(title, ' :')
        print('the optimum weight vector is: ')
        print(self.sol)
        return

    def plot_err(self, title='Square error', logy=0, logx=0):
        err = self.err
        plt.figure()
        if (logy == 0) & (logx == 0):
            plt.plot(err[:, 0], err[:, 1])
        if (logy == 1) & (logx == 0):
            plt.semilogy(err[:, 0], err[:, 1])
        if (logy == 0) & (logx == 1):
            plt.semilogx(err[:, 0], err[:, y])
        if (logy == 1) & (logx == 1):
            plt.loglog(err[:, 0], err[:, 1])
        plt.xlabel('n')
        plt.ylabel('e(n)')
        plt.title(title)
        plt.margins(0.01, 0.1)  # leave some space
        plt.grid()
        plt.show()
        return


class SolveLLS(SolveMinProb): #class SolveLLS belongs to class SolveMinProbl
    def run(self): #sets the values of self.sol(w_hat) and self.min(the error square norm)
        A=self.matr
        y=self.vect
        w_hat=np.linalg.inv(A.T@A)@(A.T@y) #w_hat is the inverse of (A.T@A)@(A.t@y); A@B is the product of the two Ndarrays
        self.sol=w_hat
        self.min=np.linalg.norm(A@w_hat-y)**2 #self.min is the quadratic norm of A@w_hat-y
        return

class SolveGrad(SolveMinProb):
    def run(self, gamma=1e-5, Nit=1000):
        self.err=np.zeros((Nit,2), dtype=float)
        self.gamma=gamma
        self.Nit=Nit #number of iterations
        A=self.matr
        y=self.vect
        w=np.random.rand(self.Nf,) #random initialization of the weight vector
        for it in range(Nit):
            grad=2*A.T@(A@w-y)
            w=w-gamma*grad
            self.err[it,0]=it
            self.err[it,1]=np.linalg.norm(A@w-y)**2
            self.sol=w
            self.min=self.err[it,1]


class SolveSteep(SolveMinProb):
    def run(self, eps, Nit):
        self.sol = np.zeros((self.Nf), dtype=float)
        self.err=np.zeros((Nit,2), dtype=float)
        self.Nit=Nit
        A=self.matr
        y=self.vect
        w=np.random.rand(self.Nf,)
        H=2*(A.T@A)
        for it in range(Nit):
            grad=2*A.T@(A@w-y)
            gamma=(np.linalg.norm(grad)**2)/(grad.T@H@grad)
            fx = np.linalg.norm(y - A @ w) ** 2
            w=w-gamma*grad
            self.err[it,0]=it
            self.err[it,1]=np.linalg.norm(A@w-y)**2
            fx1 = np.linalg.norm(y - A @ w)**2
            if np.abs(fx1-fx) < eps:
                #print(f'{eps}: {it}')
                self.n = it
                break
        self.sol=w
        self.min=self.err[it,1]


    def getWHat(self):
        return self.sol
    
    def getNit(self):
        return self.n

    def predict(self, X_te):
        y_hat_tr = self.matr @ self.sol
        y_hat_te = X_te @ self.sol
        return y_hat_tr, y_hat_te

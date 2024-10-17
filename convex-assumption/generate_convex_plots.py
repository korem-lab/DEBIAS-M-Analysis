import numpy as np
import pandas as pd
from debiasm.torch_functions import rescale
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
sys.path.append('../v1-DEBIAS-M-Analysis/General_functions/')
import plotting

def main():

    np.random.seed(42)
    X = np.random.rand(4, 3)
    bcfs = np.ones((4, 3))

    x1 = np.logspace(-5, 5, 101, base=2)
    x2 = np.logspace(-5, 5, 101, base=2)
    X1,X2 = np.meshgrid(x1,x2)


    def test_bcfs(x1, x2):
        bcfs_tmp=bcfs.copy()
        bcfs_tmp[:,0]=x1
        bcfs_tmp[:,1]=x2

        X_new = rescale( X * bcfs_tmp)

        ## getting and comparing the batch averages
        X_new[1, :] = X_new[0] + X_new[1]
        X_new[2, :] = X_new[2] + X_new[3]


        X_new = rescale(X_new[1:3])
        return( pairwise_distances(X_new
                    )[np.triu( np.ones((X_new.shape[0], 
                                        X_new.shape[0]))>0, k=1)].sum() )


    Z = np.zeros(X1.shape) 
    for i in range(X1.shape[1]):
        for j in range(X2.shape[0]):
            Z[j,i] = test_bcfs(X1[i,j], X2[i,j])

    fig, ax = plt.subplots(figsize=(8,8))
    im=plt.pcolormesh(X1,
                      X2,
                      Z,
                      cmap='RdBu'
                      )

    fig.colorbar(im)

    plt.loglog()
    plt.xlabel("Taxon #1's BCF for both studies")
    plt.ylabel("Taxon #2's BCF for both studies")
    plt.title('Similar point, but analyzing the bcfs directly\n'+\
              'in multiplicative space (as opposed to logspace)\n'+\
             'can still see straight lines between "dark-red" areas\n'+\
              'that pass through blue/white regions\n'+\
              'This plot is in loglog space'
             )

    def label_point(x, y, val, ax):
        a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
        for i, point in a.iterrows():
            if i==2:
                ax.text(point['x']+0.25, 
                        point['y']-0.25, 
                        str(point['val']), 
                        color='gold', 
                        fontweight='bold'
                       )

            else:
                ax.text(point['x']+point['x']/3, 
                        point['y']-point['y']/3, 
                        str(point['val']), 
                        color='gold', 
                        fontweight='bold'
                       )
    p1=[ 0.1, 0.1 ]
    p2=[ 4.5 , 4.5 ]


    sns.scatterplot([p1[0], p2[0], (p1[0] + p2[0])/7 ], 
                    [p1[1], p2[1], (p1[1] + p2[1])/7 ], 
                     color='gold', 
                    s = 1000, 
                    ax=ax,
                   )
    plt.plot([p1[0], p2[0]], 
                 [p1[1], p2[1]], 
                 color='gold', 
                 linewidth = 5,
                 )

    label_point(pd.Series([p1[0], p2[0], (p1[0] + p2[0])/7]), 
                pd.Series([p1[1], p2[1], (p1[1] + p2[1])/7]),
                pd.Series(['$A$', '$B$', '$\dfrac{A+B}{7}$'
                          ]),
                plt.gca()
               ) 


    plt.savefig('../plots/convex-assumptions/3_taxa_heatmap.pdf', 
                dpi=900, 
                bbox_inches='tight', 
                format='pdf'
                )



    np.random.seed(42)
    X = np.random.rand(4, 2)
    bcfs = np.zeros((4, 2))

    x1 = np.linspace(-50, 50, 101)
    x2 = np.linspace(-50, 50, 101)
    X1,X2 = np.meshgrid(x1,x2)


    def test_bcfs(x1, x2):
        bcfs_tmp=bcfs.copy()
        bcfs_tmp[:2,0]=x1
        bcfs_tmp[2:,0]=x2

        X_new = rescale( X * np.power(2, bcfs_tmp) )

        ## sums of each 2-sample batch
        X_new[1, :] = X_new[0] + X_new[1]
        X_new[2, :] = X_new[2] + X_new[3]


        X_new = rescale(X_new[1:3])
        return( pairwise_distances(X_new
                    )[np.triu( np.ones((X_new.shape[0], 
                                        X_new.shape[0]))>0, k=1)].sum() )

    Z = np.zeros(X1.shape) 
    for i in range(X1.shape[1]):
        for j in range(X2.shape[0]):
            Z[j,i] = test_bcfs(X1[i,j], X2[i,j])

    plt.figure(figsize=(8,8))
    ax=sns.heatmap(Z, cmap='RdBu', 
                   xticklabels=x1, 
                   yticklabels=x2
                  )

    sns.scatterplot([10, 65, (10 + 65)/2], 
                     [40, 90, (40+90)/2], 
                    color='gold', 
                    s = 1000, 
                   ax=ax, 
                   )
    sns.lineplot([10, 65], 
                 [40, 90], 
                color='gold', 
                linewidth = 5, ax=ax)


    plt.xlabel("Taxon #1's BCF for study #1")
    plt.ylabel("Taxon #1's BCF for study #2")
    plt.title('Pairwise cross-study loss across\na range of BCF parameters\n'+\
              'For a synthetic dataset of 4 samples, 2 studies, 2 taxa\n'
              "The gold points illustrate that this space is not convex,\n"+\
              "since for the enpoints $A,B$, and the center point $\dfrac{A+B}{2}$:\n"
              "$f( \dfrac{A+B}{2}) > \dfrac{ f(A) + f(B) }{2}$\n"
             )
    plt.xticks(ticks=[0, 25, 50, 75, 100],
               labels=[-50, -25, 0, 25, 50])
    plt.yticks(ticks=[0, 25, 50, 75, 100],
               labels=[-50, -25, 0, 25, 50])


    def label_point(x, y, val, ax):
        a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
        for i, point in a.iterrows():
            if i==2:
                ax.text(point['x']-20, 
                        point['y']+11.5, 
                        str(point['val']), 
                        color='gold', 
                        fontweight='bold'
                       )

            else:
                ax.text(point['x']+4, 
                        point['y']-4, 
                        str(point['val']), 
                        color='gold', 
                        fontweight='bold'
                       )

    label_point(pd.Series([10, 65, (10 + 65)/2]), 
                pd.Series([40, 90, (40+90)/2]),
                pd.Series(['$A$', '$B$', '$\dfrac{A+B}{2}$'
                          ]),
                plt.gca()) 

    plt.savefig('../plots/convex-assumptions/2_taxa_heatmap.pdf', 
                dpi=900, 
                bbox_inches='tight', 
                format='pdf'
                )

if __name__=='__main__':
    main()



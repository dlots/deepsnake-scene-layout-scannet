# В данный момент метрики находятся в разработке

    #Количество правильных 2d сегментов
def correct2d_num(numseg,numdot,X,Y,R)
    correct=0
    flag=0
    for k in range(numseg):
        for i in range(numdot):
            Xcenter+=X[i]
            Ycenter+=Y[i]
        Xcentralcord=Xcenter/numdot
        Ycentralcord = Ycenter/numdot
        for i in range(numdot):
            jx=sqrt(sqr(X[i]-Xcentralcord))
            jy = sqrt(sqr(Y[i] - Ycentralcord))
        if (jx>R) or (jy>R):
            flag=1
        if flag==0:
            correct+=1
        flag=0
        Xcenter=0
        Ycenter=0
    return correct


    #Нахождение лучших сегментов для золотых
def myf(numgs,numseg,GSeg,Seg,x,y,xg,yg)
    a = [-1]*numgs
    for i in range(numgs):
        mindist=R
        for j in range(numseg):
            if x[j]!=-1: #x[j]=-1 у удаленного сегмента
                dist=sqrt(sqr(x[j]-xg[j])+sqr(y[j]-yg[j])) #расстояние между конечными точками
                if mindist>dist:
                    mindist=dist
                    imindist=j
        a[i]=imindist #массив в котором i номер золотого сегмента передается значение равное сопоставимому ему прогнозируемого
        x[imidinst]=123

    # number of false positives
    countfalsepos=0
    for i in range(numseg):
        if x[i]==123:
            countfalsepos++
    # number correct layouts
    flag = 0
    for i in range(numgs):
        if (a[i]==-1):
            flag = 1
    return a,countfalsepos,flag

def accuracy()
    numcorlayouts=0
    for i in range(numlayouts):
        if flag(myf())==0:
            numcorlayouts+=1
    acc = numcorlayouts/numlayouts
    return acc






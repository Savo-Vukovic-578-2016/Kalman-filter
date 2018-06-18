 
import pylab
import math
import random
import numpy


N = 100 #broj iteracija
dt = 0.1 #vremenski interval
a = 2
#inputs


N = 100
visina = []
brzina = []
visina.append(4000)
brzina.append(280)
for i in range(1,N):
    visina.append(visina[i-1] + 50)
    brzina.append(brzina[i-1] + 2)

praveVr = numpy.zeros(shape=(N,2))
merenja = numpy.zeros(shape=(N,2))
for i in range(0,N):
    praveVr[i][0] = visina[i]
    praveVr[i][1] = brzina[i]
    merenja[i][0] = visina[i] + random.gauss(0,400)
    merenja[i][1] = brzina[i] + random.gauss(0,100) 

A = numpy.array([ [1,dt] , [0,1] ])#state matrix
B = numpy.array([  [0.5 * (dt ** 2)] , [dt] ])#control matrix
u = numpy.array([2]) #control value
H = numpy.eye(2)#observation matrix
_x= 4000 #initial state estimate
_P= numpy.array([      [400,0] , [0,25]    ])#initial coveriance estimate
Q = 0#error in process
R =numpy.array([ [800,0],  [0,1000]    ]) #error in measurment

def pr_kalman(A1,B1,X1,u1,P,Q1):
    Xp = A1 @ X1 + B1 @ u1 #predicted estimate
    Pp = (A1 @ P) @ numpy.transpose(A1) + Q1 #predicted prob. estimate
    #Pp[0][1] = 0
    #Pp[1][0] = 0    
    
    return(Xp,Pp)
    
def up_kalman(H1,Pp1,Xp1,R1,Yp1):
    inn = H1 @ Pp1 @ numpy.transpose(H1) + R1
    #inn2 = Yp1 - H1 * Xp1    
    K = Pp1 @ numpy.transpose(H1) @ numpy.linalg.inv(inn) #kalman gain    
    Xk = Xp1 + K @ (Yp1 - H1 @ Xp1) #new estimate
    Pk = (numpy.eye(2) - K @ H1) @ Pp1 #new predicted problem estimate
    return(Xk,Pk)
    
output = numpy.zeros(shape = (N,2))
Xx = numpy.array([[merenja[0][0]],[merenja[0][1]]])

for i in range(1,N):
    (Xp,Pp) = pr_kalman(A,B,Xx,u,_P,Q)
    Y = numpy.array([ [merenja[i][0]] , [merenja[i][1]]  ])
    (Xk,Pk) = up_kalman(H,Pp,Xp,R,Y)
    
    output[i][0] = Xk[0][0]
    output[i][1] = Xk[1][0]

    Xx = Xk
    _P = Pk #update the values



ploting = []
mer = []
mes = []

plt1 = []
mer1 = []
mes1 = []

for i in range(0,N):
    ploting.append(output[i][0])
    mer.append(merenja[i][0])
    mes.append(praveVr[i][0])
    plt1.append(output[i][1])
    mer1.append(merenja[i][1])
    mes1.append(praveVr[i][1])
    
    
pylab.plot(mer,'+',ploting,'r--',mes,'g-')
pylab.xlabel('X position')
pylab.ylabel('Y position')
pylab.title('Avion u letu')
pylab.legend(('measured','Kalman','true'))
pylab.show()
pylab.figure(2)
pylab.plot(plt1,'r',mer1,'r+',mes1,'g')
pylab.legend(('kalman','measured','true'))








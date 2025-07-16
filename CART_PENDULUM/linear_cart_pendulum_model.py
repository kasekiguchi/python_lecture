def Ac_CartPendulum(in1):
    Dp = in1[4]
    Dth = in1[5]
    J = in1[2]
    M = in1[1]
    gravity = in1[6]
    lg = in1[3]
    m = in1[0]
    t2 = J*m
    t3 = M+m
    t4 = lg**2
    t5 = J*M
    t6 = M*m*t4
    t7 = t2+t5+t6
    t8 = 1.0/t7
    # return [[0.0,0.0,0.0,0.0],[0.0,0.0,-gravity*m**2*t4*t8,gravity*lg*m*t3*t8],[1.0,0.0,-Dp*t8*(J+m*t4),Dp*lg*m*t8],[0.0,1.0,Dth*lg*m*t8,-Dth*t3*t8]]
    return [[ 0.0,  0.0,   1.0, 0.0],
 [ 0.0,  0.0,   0.0, 1.0],
 [ 0.0, -gravity*m**2*t4*t8, -Dp*t8*(J+m*t4), Dth*lg*m*t8],
 [ 0.0,  gravity*lg*m*t3*t8, Dp*lg*m*t8,     -Dth*t3*t8]]
def Bc_CartPendulum(in1):
    J = in1[2]
    M = in1[1]
    a = in1[7]
    lg = in1[3]
    m = in1[0]
    t2 = J*m
    t3 = lg**2
    t4 = J*M
    t5 = M*m*t3
    t6 = t2+t4+t5
    t7 = 1.0/t6
    return [[0.0],[0.0],[a*t7*(J+m*t3)],[-a*lg*m*t7]]


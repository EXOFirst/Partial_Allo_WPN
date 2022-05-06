"""
This file is system environment for agent.py.
"""

import numpy as np

class UAV():
    '''
    <Constants AND Variables>
        H: UAV flying height (m)
        N: number of users
        C: service region radius (m)
        T: operation period of the UAV (time slots)
        ts: time slot duration (seconds)
        Eb: UAV energy budget per slot (J)
        Pf: flying power of the UAV (watts)
        Ph: hovering power of the UAV (watts)
        v: UAV velocity (m/s)
        
        iota: chip energy consumption coefficient (kappa also same)
        F: number of UAV CPU cycle per bit 
        FI: number of GUs CPU cycle per bit
        Fil: computing capacity of GUs
        Fus: computing capacuty of UAV
        D: input data size (bits)

        B: system bandwidth (MHz)
        beta: channel power gain at reference distance d0 = 1 m (dB)
        alpha: path-loss exponent
        sigma: noise power spectral density (dBm/Hz)
        
        Emax: battery capacity of the user, (mJ)
        mu: average harvested energy (mJ)

        eta = energy conversion efficiency 
        ppb = transmit power at the Power Beacon (watts)

        wis: ground user(GU) locate
        tau: harvest time
        pis: GU transmit power (mW)

        w: weight (it will change(just simulate it))
        flis: computing capacity of GUs ()
        O: range of offlaoding ratio 

    '''
    def __init__(self, H=20, N=10, T=100, ts=4, Eb=8, v=15, Pf=2, Ph=1, iota=1e-27, 
                 FI=[5,25], F=[5,25], D=[0.4,1], B=30, 
                 beta=10, alpha=2, sigma=-174, Emax=10, eta=0.8, ppb = 4, pis=[1.0,10.0], w = 0.1, O =[0,1]):   

                 self.H = H
                 self.N = N
                 self.T = T
                 self.ts = ts
                 self.Eb = Eb
                 self.Pf = Pf
                 self.Ph = Ph
                 self.v = v

                 self.iota = iota
                 self.FI = np.random.uniform(FI[0],FI[1],self.N)
                 self.F = np.random.uniform(F[0],F[1],self.N)
                 self.D = np.array(D)* (10**6)

                 self.B = B*1e6
                #  self.pn = np.array(pn) *1e-3
                 self.beta = beta**(-30/10)
                 self.alpha = alpha
                 self.sigma = 10**(sigma/10)*1e-3

                 self.Es = [0.1,Emax]
                 self.Eh = self.rf_eh()
                 self.alpha = 0.1

                 self.eta = eta
                 self.ppb = ppb
                 self.tau = self.T*0.9
                 self.pis = np.random.uniform(pis[0],pis[1],self.N) * 1e-3
                 
                 self.wis = np.random.rand(self.N, 2)   
                 self.wo = np.array([0,0])  
                 self.wb = np.array([9,9])

                 self.w = w
                 self.fli = np.random.uniform(0.5,1,self.N)
                 self.O = np.array(O)
                          
    '''
    wo: UAV locate
    wb: power beacon locate
    ois_1: offloading ratio on time slot s-1 for state (uniform dist.)
    dis: inputdata of user on time slot s (uniform dist.)
    eus: UAV energy consumption on time slot s, will change (uniform dist.)
    '''
    def reset(self):
        ois_1 = np.random.uniform(self.O[0],self.O[1],self.N) 
        Dis = np.random.uniform(self.D[0],self.D[1],self.N) # change the size 
        Eus = np.random.rand(self.N) # also
        state = [ois_1, Dis, Eus] # STATE

        return state 
    
    '''
    dis: distance between UAV and GU 
    gis: channel gain between UAV and GU
    ris: data rate between user and UAV
    Fi: CPU cycle about input data
    fli: local computation capa. 
    Elc: energy consumption of local(GU)
    tlc: execution time from GU in time slot 
    Et: energy consumption for transmission
    ttis: local execution time at GU in time slot
    '''
    def local(self, ois_1, Dis):
        Fi = np.random.uniform(self.F[0],self.F[1],self.N)
        dis = np.sqrt(self.H**2 + np.sum((self.wo-self.wis)**2))
        gis = self.beta / (dis**self.alpha)
        ris = self.B * np.log2(1+ ((self.pis*gis)/self.sigma**2))
        
        Elc = self.iota * (self.fli**2) * Fi * (1-ois_1)*Dis
        tlc = ((1-ois_1)*Dis) / self.fli
        
        Et = (self.pis * (1-ois_1)*Dis) /ris
        ttis = ((1-ois_1)*Dis)/ris

        Elis = Elc + Et
        tlis = tlc + ttis
        return Elis, tlis

    '''
    Eexe: energy consumption at the UAV for processing the data 
    texe: execution time at the UAV for processing the offloaded data
    '''
    def computing(self, fus, ois_1, Dis):
        F = np.random.uniform(self.F[0], self.F[1], self.N)
        Eexe = self.iota*((fus)**2)*self.F*(ois_1*Dis) 
        texe = (ois_1 * Dis) / fus
        return Eexe, texe
        # Eexe = self.iota*((fus)^2)*self.F 
        # texe = (ois_1 * Dis) / fus
        # return Eexe, texe
    
    '''
    db2u: distace between the UAV and power beacon
    gpb: channel gain between the UAV and power beacon
    Eh: harvested energy  
    '''
    def rf_eh(self):
        db2u = np.sqrt(self.H**2 + np.sum((self.wo-self.wb)**2))
        gpb = self.beta / (db2u**self.alpha)
        Eh = self.eta * self.ppb * gpb * self.tau
        return Eh

    '''
    State = [ois_1, dis, eus]
    Action = [pis, fus]
    '''
    def step(self, state, action, time_step):
        # extract information from state
        ois_1 = state[0] # * (self.)
        Dis = state[1] * (self.D[1]-self.D[0]) + self.D[0]
        taut = np.sqrt(np.sum(()))
        Eus = state[2] * self.Pf + self.Ph  

        # extract information from state
        # pis = action[0] * (self.pis[1]-self.pis[0]) + self.pis[0]
        fus = action[1] * (self.F[1]-self.F[0]) + self.F[0]
        fli = action[2] * (self.)
        ois = action[2] 


        # need adjust Fi  
        Fi = self.F

        Elis, tlis = self.local(Fi, ois_1, Dis)
        Eexe, texe = self.computing(fus, ois_1, Dis)

        # --------------- reward ------------------  
        reward = np.sum(Elis + Eexe) + self.w*abs(tlis-self.ts) + self.w * abs(texe-self.ts)
        
        # ---------------  update state --------------- 
        ois_nxt = np.random.rand(self.N)  
        Dis_nxt = np.random.rand(self.N)
        Eus_nxt = Eus - np.sum(Eexe) + self.Eh
        # Eh 

        next_state = [ois_nxt, Dis_nxt, Eus_nxt]
        # update state
        return reward, next_state

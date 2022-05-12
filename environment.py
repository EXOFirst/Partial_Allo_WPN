import numpy as np
import os
import tensorflow as tf
from itertools import cycle
from tensorflow import keras
from collections import deque
from tensorflow.keras import layers
import numpy as np


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


# ------------------------------------environment----------------------------
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
        FU: number of UAV CPU cycle per bit (bits)
        FI: number of GUs CPU cycle per bit (bits)
        Fil: computing capacity of GUs
        Fus: computing capacuty of UAV -> 1Ghz
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

    def __init__(self, H=20, N=10, T=40, ts=6, Eb=8, v=15, Pf=2, Ph=1, iota=1e-27, FI=[1, 5], FU=[20, 30], D=[0.4, 1], B=30,
                 beta=30, alpha=2, sigma=-174, Emax=10, eta=1e-28, ppb=4, pis=[1.0, 10.0], w=0.1, O=[0, 1]):
        self.H = H
        self.N = N
        self.T = T
        self.ts = ts
        self.Eb = Eb
        self.Pf = Pf
        self.Ph = Ph
        self.v = v

        self.iota = iota
        self.FI = np.array(FI)*1e5
        self.FU = np.array(FU)*1e5
        self.D = np.array(D) * (10**6)

        self.B = B*1e6
       #  self.pn = np.array(pn) *1e-3
        self.beta = beta**(-30/10)
        self.alpha = alpha
        self.sigma = 10**(sigma/10)*1e-3

        self.Es = [0.1, Emax]

        self.alpha = 0.1

        self.eta = eta
        self.ppb = ppb
        self.tau = self.T*0.9
        self.pis = np.random.uniform(pis[0], pis[1], self.N) * 1e-3

        self.wis = np.random.rand(self.N, 2)
        self.wo = np.array([0, 0])
        self.wb = np.array([9, 9])

        self.w = w
        self.O = np.array(O)

        self.Eh = self.rf_eh()

    '''
    wo: UAV locate
    wb: power beacon locate
    ois_1: offloading ratio on time slot s-1 for state (uniform dist.)
    dis: inputdata of user on time slot s (uniform dist.)
    eus: UAV energy consumption on time slot s, will change (uniform dist.)
    '''

    def reset(self):
        ois_1 = np.random.uniform(self.O[0], self.O[1], self.N)
        # print(f'in reset ois_1: {ois_1} \n')
        Dis = np.random.uniform(
            self.D[0], self.D[1], self.N)  # change the size
        # print(f'in reset Dis{Dis} \n ')
        Eus = np.random.rand()
        state = [ois_1, Dis, Eus]  # STATE

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

    def local(self, fli, ois_1, Dis):
        Fi = np.random.uniform(self.FI[0], self.FI[1], self.N)
        dis = np.sqrt(self.H**2 + np.sum((self.wo-self.wis)**2))
        gis = self.beta / (dis**self.alpha)
        ris = self.B * np.log2(1 + ((self.pis*gis)/self.sigma**2))

        # print(f'in local \n fli shape:{np.shape(fli)} \n ois_1 shape:{np.shape(ois_1)} \n Dis shape:{np.shape(Dis)} \n')

        Elc = self.iota * (fli**2) * Fi * (1-ois_1)*Dis
        tlc = ((1-ois_1)*Dis) / fli

        Et = (self.pis * (1-ois_1)*Dis) / ris
        ttis = ((1-ois_1)*Dis)/ris

        Elis = Elc + Et
        tlis = tlc + ttis
        return Elis, tlis

    '''
    Eexe: energy consumption at the UAV for processing the data 
    texe: execution time at the UAV for processing the offloaded data
    '''

    def computing(self, fus, ois_1, Dis):
        FU = np.random.randint(self.FU[0], self.FU[1], dtype='int64')

        # print(f'in computing \n fus shape: {np.shape(fus)} \n ois_1 shape: {np.shape(ois_1)} \n Dis shape: {np.shape(Dis)}, \n FU: {np.shape(self.FU)} \n')
        # print(f'FU rand int {self.FU[0], self.FU[1]} \n')

        Eexe = self.iota*((fus)**2)*FU*(ois_1*Dis)
        texe = (ois_1 * Dis) / fus
        return Eexe, texe
        # Eexe = self.iota*((fus)^2)*self.FU
        # texe = (ois_1 * Dis) / fus
        # return Eexe, texe

    '''
    db2u: distace between the UAV and power beacon
    gpb: channel gain between the UAV and power beacon
    Eh: harvested energy  
    '''

    def rf_eh(self):
        db2u = np.sqrt(self.H**2 + np.sum((self.wo - self.wb)**2))
        gpb = self.beta / (db2u**self.alpha)
        Eh = self.eta * self.ppb * gpb * self.tau
        return Eh

    '''
    State = [ois_1, dis, eus]
    Action = [pis, fus]
    '''

    def step(self, state, action):
        # will check this env.
        # extract information from state
        ois_1 = state[0]  # first state is randomly
        Dis = state[1] * (self.D[1]-self.D[0]) + self.D[0]
        Eus = state[2]

        # extract information from state
        # pis = action[0] * (self.pis[1]-self.pis[0]) + self.pis[0]
        fus = action[0] * (self.FU[1]-self.FU[0]) + self.FU[0]
        fli = action[1] * (self.FI[1]-self.FI[0]) + self.FI[0]
        ois = action[2]

        # need adjust Fi
        # Fi = self.FU
        # fli = self.

        Elis, tlis = self.local(fli, ois_1, Dis)
        Eexe, texe = self.computing(fus, ois_1, Dis)

        # --------------- reward ------------------
        alarm = (tlis < self.ts) | (texe < self.ts)
        # print(f'in step alarm {alarm}')
        tlis, texe = tlis[alarm], texe[alarm]

        # print(f'in step raward sum(Elis+Eexe) {np.sum(Elis + Eexe)} \n w*tlis {self.w * abs(tlis-self.ts)} \n w*texe {self.w * abs(texe-self.ts)} \n')
        # print(f'in step shapes about \n Elis {np.shape(Elis)} \n Eexe {np.shape(Eexe)} \n  tlis {np.shape(tlis)}  \ texe {np.shape(texe)}')

        reward = np.sum(Elis + (Eexe)) + self.w * np.sum(abs(tlis -
                                                             self.ts)) + self.w * np.sum(abs(texe-self.ts))

        # print(f'in step reward {reward} \n')
        # ---------------  update state ---------------
        ois_nxt = ois
        Dis_nxt = np.random.uniform(self.D[0], self.D[1], self.N)
        # print(f'in step Dis_nxt: {Dis_nxt} \n')
        Eus_nxt = Eus - np.sum(Eexe) + self.Eh
        Eus_nxt = np.clip(Eus_nxt, self.Es[0], self.Es[1])
        Eus_nxt = (Eus_nxt - self.Es[0]) / (self.Es[1]-self.Es[0])
        # Eh

        next_state = [ois_nxt, Dis_nxt, Eus_nxt]
        # update state
        return reward, next_state

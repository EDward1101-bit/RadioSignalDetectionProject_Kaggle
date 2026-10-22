import torch
import torch.nn as nn
import torch.nn.functional as F


class AtentieCanale(nn.Module):
    # pentru fiecare canal. Daca un filtru prinde doar zgomot, va primi o pondere 
    # aproape de 0, fiind efectiv ignorat mai departe.
    def __init__(self, nr_canale, factor=16):
        super().__init__()
        dim_ascunsa = max(nr_canale // factor, 4)
        self.atentie = nn.Sequential(
            nn.Linear(nr_canale, dim_ascunsa),
            nn.ReLU(inplace=True),
            nn.Linear(dim_ascunsa, nr_canale),
            nn.Sigmoid() 
        )

    def forward(self, x):
        lot, canale, _, _ = x.size()
        context = F.adaptive_avg_pool2d(x, 1).view(lot, canale)
        ponderi = self.atentie(context).view(lot, canale, 1, 1)
        return x * ponderi


class BlocRezidual(nn.Module):
    # Adaugarea conexiunii reziduale previne degradarea gradientului
    def __init__(self, canale_in, canale_out, stride=1):
        super().__init__()
        self.flux = nn.Sequential(
            nn.Conv2d(canale_in, canale_out, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(canale_out),
            nn.PReLU(),
            nn.Conv2d(canale_out, canale_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(canale_out)
        )
        self.atentie = AtentieCanale(canale_out)

        self.scurtatura = nn.Sequential()
        if stride != 1 or canale_in != canale_out:
            self.scurtatura = nn.Sequential(
                nn.Conv2d(canale_in, canale_out, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(canale_out)
            )

        self.activare = nn.PReLU()

    def forward(self, x):
        iesire = self.atentie(self.flux(x))
        return self.activare(iesire + self.scurtatura(x))


class ReteaCNN(nn.Module):
    def __init__(self, numar_clase=5):
        super().__init__()
        # Filtrul initial asimetric (5, 3). L-am ales pentru ca spectrogramele
        # sunt mult mai inalte pe axa de frecventa (128) decat late pe axa de timp (55).
        self.intrare = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 3), padding=(2, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.corp = nn.Sequential(
            BlocRezidual(32, 64, stride=1),
            BlocRezidual(64, 64, stride=1),
            BlocRezidual(64, 128, stride=2),
            BlocRezidual(128, 128, stride=1),
            BlocRezidual(128, 256, stride=2),
            BlocRezidual(256, 256, stride=1),
            BlocRezidual(256, 256, stride=2)
        )

        self.agregare = nn.AdaptiveAvgPool2d((1, 1))
        self.clasificator = nn.Linear(256, numar_clase)

    def forward(self, x):
        x = self.intrare(x)
        x = self.corp(x)
        x = self.agregare(x).view(x.size(0), -1) 
        return self.clasificator(x)
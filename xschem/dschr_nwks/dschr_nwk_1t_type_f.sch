v {xschem version=3.4.1 file_version=1.2
}
G {}
K {}
V {}
S {}
E {}
N -20 -290 -20 -270 {
lab=bot}
N -70 -330 -50 -330 {
lab=bot}
N -70 -330 -70 -270 {
lab=bot}
N -70 -270 20 -270 {
lab=bot}
N 20 -330 20 -270 {
lab=bot}
N 10 -330 20 -330 {
lab=bot}
N -90 -270 -70 -270 {
lab=bot}
N -20 -390 -20 -330 {
lab=top}
N -90 -390 -20 -390 {
lab=top}
C {devices/iopin.sym} -90 -390 2 0 {name=p6 lab=top}
C {devices/iopin.sym} -90 -270 2 0 {name=p7 lab=bot}
C {sky130_fd_pr/nfet_01v8.sym} -20 -310 3 0 {name=M1
L='L'
W='W'
nf=1 
mult=1
ad="'int((nf+1)/2) * W/nf * 0.29'" 
pd="'2*int((nf+1)/2) * (W/nf + 0.29)'"
as="'int((nf+2)/2) * W/nf * 0.29'" 
ps="'2*int((nf+2)/2) * (W/nf + 0.29)'"
nrd="'0.29 / W'" nrs="'0.29 / W'"
sa=0 sb=0 sd=0
model=nfet_01v8
spiceprefix=X
}

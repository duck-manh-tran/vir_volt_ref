v {xschem version=3.4.1 file_version=1.2
}
G {}
K {}
V {}
S {}
E {}
N 40 -80 40 -60 {
lab=top}
N -20 -30 -0 -30 {
lab=bot}
N 40 -30 60 -30 {
lab=top}
N 60 -60 60 -30 {
lab=top}
N 40 -60 60 -60 {
lab=top}
N -20 -30 -20 -0 {
lab=bot}
N -20 30 40 30 {
lab=bot}
N -40 30 -20 30 {
lab=bot}
N -40 -80 40 -80 {
lab=top}
N 40 0 40 30 {
lab=bot}
N -20 -0 -20 30 {
lab=bot}
C {devices/iopin.sym} -40 -80 2 0 {name=p6 lab=top}
C {devices/iopin.sym} -40 30 2 0 {name=p7 lab=bot}
C {sky130_fd_pr/nfet_01v8.sym} 20 -30 0 0 {name=M2
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

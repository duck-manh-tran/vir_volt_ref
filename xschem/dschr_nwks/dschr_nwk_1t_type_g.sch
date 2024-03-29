v {xschem version=3.4.1 file_version=1.2
}
G {}
K {}
V {}
S {}
E {}
N -20 -30 -0 -30 {
lab=top}
N 40 -0 40 20 {
lab=bot}
N -20 -60 -20 -30 {
lab=top}
N 40 -30 50 -30 {
lab=bot}
N 50 -30 50 20 {
lab=bot}
N -30 20 50 20 {
lab=bot}
N 40 -80 40 -60 {
lab=top}
N -30 -80 40 -80 {
lab=top}
N -20 -80 -20 -60 {
lab=top}
N -40 -80 -30 -80 {
lab=top}
N -40 20 -30 20 {
lab=bot}
N -50 -80 -40 -80 {
lab=top}
C {devices/iopin.sym} -50 -80 2 0 {name=p6 lab=top}
C {devices/iopin.sym} -40 20 2 0 {name=p7 lab=bot}
C {sky130_fd_pr/nfet_01v8.sym} 20 -30 0 0 {name=M1
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

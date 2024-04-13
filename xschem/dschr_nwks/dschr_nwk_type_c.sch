v {xschem version=3.4.1 file_version=1.2
}
G {}
K {}
V {}
S {}
E {}
N 20 -200 20 -180 {
lab=bot}
N 20 -200 60 -200 {
lab=bot}
N 60 -200 60 -130 {
lab=bot}
N 20 -130 60 -130 {
lab=bot}
N 20 -140 20 -130 {
lab=bot}
N 20 -130 20 -80 {
lab=bot}
N -40 -140 -10 -140 {
lab=#net1}
N 20 -80 20 -70 {
lab=bot}
N 50 -140 80 -140 {
lab=top}
N 20 -70 80 -70 {
lab=bot}
N -40 -70 20 -70 {
lab=bot}
N -110 -140 -40 -140 {
lab=#net1}
N -110 -80 -110 -70 {
lab=bot}
N -110 -70 -40 -70 {
lab=bot}
C {devices/iopin.sym} 80 -140 0 0 {name=p6 lab=top}
C {devices/iopin.sym} 80 -70 0 0 {name=p7 lab=bot}
C {sky130_fd_pr/nfet_01v8.sym} 20 -160 1 0 {name=M1
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
C {devices/capa.sym} -110 -110 0 0 {name=C2
m=1
value=1p
footprint=1206
device="ceramic capacitor"}

v {xschem version=3.4.1 file_version=1.2
}
G {}
K {}
V {}
S {}
E {}
N 30 -10 30 30 {
lab=#net1}
N -30 -40 -30 60 {
lab=bot}
N 30 -40 40 -40 {
lab=bot}
N 30 -90 30 -70 {
lab=top}
N 30 60 40 60 {
lab=bot}
N 40 -40 40 60 {
lab=bot}
N -30 -40 -10 -40 {
lab=bot}
N -30 60 -10 60 {
lab=bot}
N -30 60 -30 100 {
lab=bot}
N -30 100 30 100 {
lab=bot}
N 30 90 30 100 {
lab=bot}
N -50 100 -30 100 {
lab=bot}
N -50 -90 30 -90 {
lab=top}
N 40 60 40 100 {
lab=bot}
N 30 100 40 100 {
lab=bot}
C {devices/iopin.sym} -50 -90 2 0 {name=p6 lab=top}
C {devices/iopin.sym} -50 100 2 0 {name=p7 lab=bot}
C {sky130_fd_pr/nfet_01v8.sym} 10 -40 0 0 {name=M1
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
C {sky130_fd_pr/nfet_01v8.sym} 10 60 0 0 {name=M2
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

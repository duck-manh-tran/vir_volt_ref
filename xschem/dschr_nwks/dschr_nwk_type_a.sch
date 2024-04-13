v {xschem version=3.4.1 file_version=1.2
}
G {}
K {}
V {}
S {}
E {}
N -230 -140 -220 -140 {
lab=bot}
N -230 -140 -230 -110 {
lab=bot}
N -230 -90 -180 -90 {
lab=bot}
N -250 -90 -180 -90 {
lab=bot}
N -180 -140 -160 -140 {
lab=bot}
N -160 -140 -160 -110 {
lab=bot}
N -180 -90 -160 -90 {
lab=bot}
N -180 -190 -180 -170 {
lab=top}
N -250 -190 -180 -190 {
lab=top}
N -230 -110 -230 -90 {
lab=bot}
N -180 -110 -180 -90 {
lab=bot}
N -160 -110 -160 -90 {
lab=bot}
C {devices/iopin.sym} -250 -190 2 0 {name=p6 lab=top}
C {devices/iopin.sym} -250 -90 2 0 {name=p7 lab=bot}
C {sky130_fd_pr/nfet_01v8.sym} -200 -140 0 0 {name=M2
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

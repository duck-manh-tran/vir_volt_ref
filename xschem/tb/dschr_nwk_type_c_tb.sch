v {xschem version=3.4.4 file_version=1.2
}
G {}
K {}
V {}
S {}
E {}
N -90 -130 -90 -60 {
lab=vp}
N -90 -130 30 -130 {
lab=vp}
N 30 -130 30 -100 {
lab=vp}
N 30 20 30 50 {
lab=GND}
N -90 50 30 50 {
lab=GND}
N -90 0 -90 50 {
lab=GND}
C {devices/gnd.sym} -30 50 0 0 {name=l1 lab=GND}
C {devices/lab_wire.sym} -30 -130 0 0 {name=p1 sig_type=std_logic lab=vp}
C {sky130_fd_pr/corner.sym} 80 -120 0 0 {name=CORNER only_toplevel=false corner=tt}
C {devices/code.sym} 190 -120 0 0 {name=ctrl_sim only_toplevel=false value="tcleval(
.include $::netlist_dir/ctrl_sims.spice
)"}
C {devices/capa.sym} -90 -30 0 0 {name=C1
m=1
value=50f
footprint=1206
device="ceramic capacitor"}
C {dschr_nwks/dschr_nwk_1t_type_c.sym} 0 0 0 0 {name=x2 L=0.15 W=0.5}

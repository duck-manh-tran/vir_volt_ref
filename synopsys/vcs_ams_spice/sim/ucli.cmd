set openFiles [split [dump -opened] "\n"]
if {([lindex $openFiles 0 0] == "No") || ([llength [lindex $openFiles 1]] < 3)} {
    dump -type fsdb -file top.fsdb
    set openFiles [split [dump -opened] "\n"]
}
set digitalDumpFile [lindex $openFiles 1 0]
dump -add / -depth 1 -fid $digitalDumpFile
dump -add / -ports -fid $digitalDumpFile

run 1ms
quit

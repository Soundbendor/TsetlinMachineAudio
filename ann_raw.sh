#!bin/bash
#!bin/bash

annotations=()

readarray fileNameArray < <(find ~+ -regex '.*/*+_[aeiou]\.wav' -print)

for f in ${!fileNameArray[@]}; do
        s=${fileNameArray[$f]}
        char=${s: -6: 1}
	# duration=$(ffprobe -i $s -show_entries format=duration -v quiet -of csv="p=0")
    
        case $char in

                i)
                        annotations+=("$s,0")
                        ;;
                e)
                        annotations+=("$s,1")
                        ;;
                a)
                        annotations+=("$s,2")
                        ;;
                o)
                        annotations+=("$s,3")
                        ;;
                u)
                        annotations+=("$s,4")
                        ;;
        esac
        

done

printf "%s\n" "${annotations[@]}" > annotations_raw.txt

exit 0

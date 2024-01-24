#!bin/bash

annotations=()

# readarray -d '' fileNameArray < <(find ~+ -name "*.wav" -print0)
readarray fileNameArray < <(find ~+ -name "*.wav" -print)

for f in ${!fileNameArray[@]}; do
        s=${fileNameArray[$f]}
        char=${s: -6: 1}
	# duration=$(ffprobe -i $s -show_entries format=duration -v quiet -of csv="p=0")
        num_char=${s: -8: 1}

        re='^[0-9]+$'
        if [[ $num_char =~ $re ]] ; then
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
        fi

done

printf "%s\n" "${annotations[@]}" > annotations.txt

exit 0

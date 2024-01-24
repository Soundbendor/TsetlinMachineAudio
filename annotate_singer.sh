#!bin/bash

annotations=()

# readarray -d '' fileNameArray < <(find ~+ -name "*.wav" -print0)
readarray fileNameArray < <(find ~+ -name "*.wav" -print)




re='(male[0-9][0-1]?|female[0-9][0-1]?)'

for f in ${!fileNameArray[@]}; do
        s=${fileNameArray[$f]}
        

        if [[ $s =~ $re ]] ; then
            #sex in BASH_REMATCH[1]
            #number in BASHREMATCH[2]
                if [[ ${BASH_REMATCH[1]} = "male"]] ; then
                    annotations+=("$s,((${BASH_REMATCH[2]}-1)),${BASH_REMATCH[1]}")
                else
                    annotations+=("$s,((${BASH_REMATCH[2]}+10)),${BASH_REMATCH[1]}")
                fi
        fi

done

printf "%s\n" "${annotations[@]}" > singer_annotations.txt

exit 0






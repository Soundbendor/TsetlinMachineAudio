#!bin/bash

annotations=()

# readarray -d '' fileNameArray < <(find ~+ -name "*.wav" -print0)
readarray fileNameArray < <(find ~+ -name "*.wav" -print)


# TODO check if the spoken is really spoken
re='(vibrato|straight|breathy|vocal_fry|lip_trill|trill|trillo|inhaled|belt|spoken)'

for f in ${!fileNameArray[@]}; do
        s=${fileNameArray[$f]}
        



        if [[ $s =~ $re ]] ; then
                case ${BASH_REMATCH[0]} in

                        vibrato)
                                annotations+=("$s,0")
                                ;;
                        straight)
                                annotations+=("$s,1")
                                ;;
                        breathy)
                                annotations+=("$s,2")
                                ;;
                        vocal_fry)
                                annotations+=("$s,3")
                                ;;
                        lip_trill)
                                annotations+=("$s,4")
                                ;;
                        trill)
                                annotations+=("$s,5")
                                ;;
                        trillo)
                                annotations+=("$s,6")
                                ;;
                        inhaled)
                                annotations+=("$s,7")
                                ;;
                        belt)
                                annotations+=("$s,8")
                                ;;
                        spoken)
                                annotations+=("$s,9")
                                ;;
                esac
        fi

done

printf "%s\n" "${annotations[@]}" > tech_annotations.txt

exit 0






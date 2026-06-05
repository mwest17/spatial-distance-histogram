fail="passed"
for n in {7..1000..5} 
do
     if ! ./a.out 10000 $n 64 | grep -q "Not different";
     then
          fail="failed"
          break
     fi

done
echo $fail
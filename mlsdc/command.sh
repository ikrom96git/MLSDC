python3 dummy_True.py
matlab -nodisplay -nosplash -nodesktop -r "run('test_poly_approx.m');exit;" | tail -n +11
for ii in 1 2 3 4
do
   python3 dummy_False.py
   matlab -nodisplay -nosplash -nodesktop -r "run('test_poly_approx.m');exit;" | tail -n +11
done

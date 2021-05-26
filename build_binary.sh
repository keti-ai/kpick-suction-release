rm -rf build
python setup.py build_ext
stubgen ketisdk/vision/detector/pick/suction/*.py -o build/lib.linux-x86_64-3.6/

rm -rf bin
mkdir bin 
rsync -r -L . bin   --exclude='ketisdk/vision/detector/pick/suction' --exclude='ketisdk/vision/detector/pick/grip' --exclude='ketisdk/vision/detector/pick/grip3' --exclude='data' --exclude='build' --exclude='bin' 
rsync -r -L build/lib.linux-x86_64-3.6/ketisdk/vision/detector/pick/suction bin/ketisdk/vision/detector/pick

rm -rf build
#cd bin
#python setup.py sdist bdist_wheel
#twine upload dist/*






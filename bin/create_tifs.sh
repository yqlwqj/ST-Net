#!/bin/bash

#cd `python -c "import histonet; print(histonet.config.SPATIAL_RAW_ROOT)"`
#for i in */*/*.jpg;
#do
#    echo ${i}
#    convert ${i} -define tiff:tile-geometry=256x256 ${i%.jpg}.tif
#done
#!/bin/bash

path="/home/ben/project"

# Search .jpg file in all directory

function readJpg ()
{
	for file in `ls $1`
	do
	if [ -d $1"/"$file ]
	then
		readJpg $1"/"$file
	else

		if echo $1"/"$file | grep -q -E '.jpg'
		then
		i="$1"/"$file"
			echo $i
			convert $i -define tiff:tile-geometry=256x256 ${i%.jpg}.tif
		fi
	fi
	done
}

readJpg $path

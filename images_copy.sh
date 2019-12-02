items=(
	"aquarius"
	"ayataka_brown"
	"coca_cola"
	"genmai"
	"koicha"
	"natural_green"
	"pokari"
	"ayataka"
	"caplis"
	"dekavita"
	"iemon"
	"namacha"
	"ooi_ocha"
	"tropicana"
)
if [ -d self_images_$1 ]; then
	echo "フォルダはすでにあります"
else
	echo "フォルダを作成します"
	mkdir self_images_$1
fi

for i in $(seq 0 $1); do
	echo "${items[${i}]}をコピー"
	cp -r self_images/${items[${i}]}/ self_images_$1/${items[${i}]}/
done

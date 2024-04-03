import csv
import sys
import glob
import os
import os.path as path
import exifread


def read_image_names(dirname):
    files = glob.glob(os.path.join(dirname, '*'))
    return sorted([os.path.basename(f) for f in files if is_image(f)])


def is_image(filename):
    return True if filename.lower().endswith(('.jpg', '.png', '.gif')) else False


def read_exif(dirname):
    exif = []
    img_files = glob.glob(os.path.join(dirname, '*'))
    sorted_img_files = sorted([f for f in img_files if is_image(f)])

    for img_file in sorted_img_files:
        with open(img_file, 'rb') as file:
            tags = exifread.process_file(file, details=False)
            exif.append(eval(str(tags['EXIF ExposureTime'])))
            # exif['iso'] = int(str(tags['EXIF ISOSpeedRatings']))
            # exif['focal_len'] = int(str(tags['EXIF FNumber']))
    return [(float)(t) for t in exif]


def read_times_from_text(filename):
    def to_float(fraction):
        if '/' in fraction:
            numerator, denominator = map(int, fraction.split('/'))
            return float(numerator / denominator)
        else:
            return float(fraction)

    with open(filename, 'r') as f:
        lines = f.readlines()

    return [to_float(line.split()[0]) for line in lines]


def create_csv(image_names, shutter_times, filename):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'shutter_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for img, t in zip(image_names, shutter_times):
            writer.writerow({'filename': img, 'shutter_time': t})


if __name__ == '__main__':
    indir = sys.argv[1:]
    txt_file = path.join(indir, 'exposure.txt')
    out_file = path.join(indir, 'exposure_times.csv')

    image_names = read_image_names(indir)
    exposure_times = read_exif(indir)

    exposure_times = read_times_from_text(
        txt_file) if path.exists(txt_file) else read_exif(indir)

    create_csv(image_names, exposure_times, out_file)

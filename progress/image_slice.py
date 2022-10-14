from module import slice

if __name__ == '__main__':
    print(f"0-多波段格网裁剪、1-单波段格网裁剪、2-多波段随机裁剪、3-单波段随机裁剪、4-制作深度学习样本-格网裁剪、5-制作深度学习样本-随机裁剪")
    slice_type = input(f"请选择：")
    if int(slice_type) == 0:
        # 参数设置
        image_path = input(f"请输入待裁剪多波段影像路径：")
        image_slice_dir = input(f"请输入结果存放路径：")
        slice_size = int(input(f"请输入裁剪块大小："))
        edge_fill = bool(int(input(f"是否进行边缘填充（0/1）：")))
        slice.multi_bands_grid_slice(image_path, image_slice_dir, slice_size, edge_fill=edge_fill)

    elif int(slice_type) == 1:
        # 参数设置
        band_path = input(f"请输入待裁剪单波段影像路径：")
        band_slice_dir = input(f"请输入结果存放路径：")
        slice_size = int(input(f"请输入裁剪块大小："))
        edge_fill = bool(int(input(f"是否进行边缘填充（0/1）：")))
        slice.single_band_grid_slice(band_path, band_slice_dir, slice_size, edge_fill=edge_fill)

    elif int(slice_type) == 2:
        # 参数设置
        image_path = input(f"请输入待裁剪多波段影像路径：")
        image_slice_dir = input(f"请输入结果存放路径：")
        slice_size = int(input(f"请输入裁剪块大小："))
        slice_count = int(input(f"请输入裁剪数量："))
        slice.multi_bands_rand_slice(image_path, image_slice_dir, slice_size, slice_count)

    elif int(slice_type) == 3:
        # 参数设置
        band_path = input(f"请输入待裁剪单波段影像路径：")
        band_slice_dir = input(f"请输入结果存放路径：")
        slice_size = int(input(f"请输入裁剪块大小："))
        slice_count = int(input(f"请输入裁剪数量："))
        slice.single_band_rand_slice(band_path, band_slice_dir, slice_size, slice_count)

    elif int(slice_type) == 4:
        # 参数设置
        image_path = input(f"请输入待裁剪多波段image路径：")
        band_path = input(f"请输入待裁剪单波段label路径：")
        image_slice_dir = input(f"请输入image裁剪结果存放路径：")
        band_slice_dir = input(f"请输入label裁剪结果存放路径：")
        slice_size = int(input(f"请输入裁剪块大小："))
        edge_fill = bool(int(input(f"是否进行边缘填充（0/1）：")))
        slice.deeplr_grid_slice(image_path, band_path, image_slice_dir, band_slice_dir, slice_size, edge_fill=edge_fill)

    elif int(slice_type) == 5:
        # 参数设置
        image_path = input(f"请输入待裁剪多波段image路径：")
        band_path = input(f"请输入待裁剪单波段label路径：")
        image_slice_dir = input(f"请输入image裁剪结果存放路径：")
        band_slice_dir = input(f"请输入label裁剪结果存放路径：")
        slice_size = int(input(f"请输入裁剪块大小："))
        slice_count = int(input(f"请输入裁剪数量："))
        slice.deeplr_rand_slice(image_path, band_path, image_slice_dir, band_slice_dir, slice_size, slice_count)

    else:
        print('输入错误')
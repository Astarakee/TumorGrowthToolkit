import SimpleITK as itk
import numpy as np
from copy import deepcopy


itk_seg = itk.ReadImage('./tissue_seg.nii.gz')
arr_seg = itk.GetArrayFromImage(itk_seg)
wm_seg = deepcopy(arr_seg)
gm_seg = deepcopy(arr_seg)
wm_seg[wm_seg !=1] = 0
gm_seg[gm_seg != 2] = 0
gm_seg[gm_seg == 2] = 1

itk_wm = itk.GetImageFromArray(wm_seg)
itk_gm = itk.GetImageFromArray(gm_seg)

def complete_itk(itk_src, itk_dst):
    itk_dst.SetOrigin(itk_src.GetOrigin())
    itk_dst.SetSpacing(itk_src.GetSpacing())
    itk_dst.SetDirection(itk_src.GetDirection())
    return itk_dst

itk_wm = complete_itk(itk_seg, itk_wm)
itk_gm = complete_itk(itk_seg, itk_gm)

itk.WriteImage(itk_wm, './wm.nii.gz')
itk.WriteImage(itk_gm, './gm.nii.gz')




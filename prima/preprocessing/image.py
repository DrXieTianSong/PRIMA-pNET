import SimpleITK as sitk

TARGET_SPACING = (1.0, 1.0, 1.0)
HU_MIN, HU_MAX = -160, 240


def output_size(image):
    size = image.GetSize()
    spacing = image.GetSpacing()
    return [int(round(sz * sp / ts))
            for sz, sp, ts in zip(size, spacing, TARGET_SPACING)]


def resample_ct(image):
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(TARGET_SPACING)
    resampler.SetSize(output_size(image))
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetInterpolator(sitk.sitkBSpline)
    resampler.SetOutputPixelType(sitk.sitkFloat32)
    out = resampler.Execute(image)
    out = sitk.Clamp(out, lowerBound=HU_MIN, upperBound=HU_MAX)
    out = sitk.Cast(out, sitk.sitkInt16)
    return out


def resample_mask(image):
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(TARGET_SPACING)
    resampler.SetSize(output_size(image))
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputPixelType(sitk.sitkUInt8)
    return resampler.Execute(image)

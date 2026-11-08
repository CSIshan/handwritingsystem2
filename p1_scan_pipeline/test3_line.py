from p1_scan_pipeline.curve_fitter2 import CurveFitter, VectorFitterParams

fitter = CurveFitter(VectorFitterParams())
curves = fitter.fit(component.mask, component.origin_offset)

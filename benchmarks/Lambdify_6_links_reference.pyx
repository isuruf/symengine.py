# Benchmark reference:
cimport numpy as cnp
import numpy as np
from libc.math cimport sin, cos

def _benchmark_reference_for_Lambdify(cnp.ndarray[cnp.float64_t] x):
    cdef cnp.ndarray[cnp.float64_t] out = np.empty(14)
    cdef double * data = <double *>out.data
    data[:] = [x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13]*x[20]*x[7]**2*cos(x[0]) + x[13]*x[21]*x[7]**2*cos(x[0]) + x[13]*x[22]*x[7]**2*cos(x[0]) + x[13]*x[23]*x[7]**2*cos(x[0]) + x[13]*x[24]*x[7]**2*cos(x[0]) + x[13]*x[25]*x[7]**2*cos(x[0]) + x[14]*x[21]*x[8]**2*cos(x[1]) + x[14]*x[22]*x[8]**2*cos(x[1]) + x[14]*x[23]*x[8]**2*cos(x[1]) + x[14]*x[24]*x[8]**2*cos(x[1]) + x[14]*x[25]*x[8]**2*cos(x[1]) + x[15]*x[22]*x[9]**2*cos(x[2]) + x[15]*x[23]*x[9]**2*cos(x[2]) + x[15]*x[24]*x[9]**2*cos(x[2]) + x[15]*x[25]*x[9]**2*cos(x[2]) + x[16]*x[23]*x[10]**2*cos(x[3]) + x[16]*x[24]*x[10]**2*cos(x[3]) + x[16]*x[25]*x[10]**2*cos(x[3]) + x[17]*x[24]*x[11]**2*cos(x[4]) + x[17]*x[25]*x[11]**2*cos(x[4]) + x[18]*x[25]*x[12]**2*cos(x[5]), -x[19]*x[13]*x[20]*cos(x[0]) - x[19]*x[13]*x[21]*cos(x[0]) - x[19]*x[13]*x[22]*cos(x[0]) - x[19]*x[13]*x[23]*cos(x[0]) - x[19]*x[13]*x[24]*cos(x[0]) - x[19]*x[13]*x[25]*cos(x[0]) + x[13]*x[14]*x[21]*x[8]**2*(-sin(x[0])*cos(x[1]) + sin(x[1])*cos(x[0])) + x[13]*x[14]*x[22]*x[8]**2*(-sin(x[0])*cos(x[1]) + sin(x[1])*cos(x[0])) + x[13]*x[14]*x[23]*x[8]**2*(-sin(x[0])*cos(x[1]) + sin(x[1])*cos(x[0])) + x[13]*x[14]*x[24]*x[8]**2*(-sin(x[0])*cos(x[1]) + sin(x[1])*cos(x[0])) + x[13]*x[14]*x[25]*x[8]**2*(-sin(x[0])*cos(x[1]) + sin(x[1])*cos(x[0])) + x[13]*x[15]*x[22]*x[9]**2*(-sin(x[0])*cos(x[2]) + sin(x[2])*cos(x[0])) + x[13]*x[15]*x[23]*x[9]**2*(-sin(x[0])*cos(x[2]) + sin(x[2])*cos(x[0])) + x[13]*x[15]*x[24]*x[9]**2*(-sin(x[0])*cos(x[2]) + sin(x[2])*cos(x[0])) + x[13]*x[15]*x[25]*x[9]**2*(-sin(x[0])*cos(x[2]) + sin(x[2])*cos(x[0])) + x[13]*x[16]*x[23]*x[10]**2*(-sin(x[0])*cos(x[3]) + sin(x[3])*cos(x[0])) + x[13]*x[16]*x[24]*x[10]**2*(-sin(x[0])*cos(x[3]) + sin(x[3])*cos(x[0])) + x[13]*x[16]*x[25]*x[10]**2*(-sin(x[0])*cos(x[3]) + sin(x[3])*cos(x[0])) + x[13]*x[17]*x[24]*x[11]**2*(-sin(x[0])*cos(x[4]) + sin(x[4])*cos(x[0])) + x[13]*x[17]*x[25]*x[11]**2*(-sin(x[0])*cos(x[4]) + sin(x[4])*cos(x[0])) + x[13]*x[18]*x[25]*x[12]**2*(-sin(x[0])*cos(x[5]) + sin(x[5])*cos(x[0])), -x[19]*x[14]*x[21]*cos(x[1]) - x[19]*x[14]*x[22]*cos(x[1]) - x[19]*x[14]*x[23]*cos(x[1]) - x[19]*x[14]*x[24]*cos(x[1]) - x[19]*x[14]*x[25]*cos(x[1]) + x[13]*x[14]*x[21]*x[7]**2*(sin(x[0])*cos(x[1]) - sin(x[1])*cos(x[0])) + x[13]*x[14]*x[22]*x[7]**2*(sin(x[0])*cos(x[1]) - sin(x[1])*cos(x[0])) + x[13]*x[14]*x[23]*x[7]**2*(sin(x[0])*cos(x[1]) - sin(x[1])*cos(x[0])) + x[13]*x[14]*x[24]*x[7]**2*(sin(x[0])*cos(x[1]) - sin(x[1])*cos(x[0])) + x[13]*x[14]*x[25]*x[7]**2*(sin(x[0])*cos(x[1]) - sin(x[1])*cos(x[0])) + x[14]*x[15]*x[22]*x[9]**2*(-sin(x[1])*cos(x[2]) + sin(x[2])*cos(x[1])) + x[14]*x[15]*x[23]*x[9]**2*(-sin(x[1])*cos(x[2]) + sin(x[2])*cos(x[1])) + x[14]*x[15]*x[24]*x[9]**2*(-sin(x[1])*cos(x[2]) + sin(x[2])*cos(x[1])) + x[14]*x[15]*x[25]*x[9]**2*(-sin(x[1])*cos(x[2]) + sin(x[2])*cos(x[1])) + x[14]*x[16]*x[23]*x[10]**2*(-sin(x[1])*cos(x[3]) + sin(x[3])*cos(x[1])) + x[14]*x[16]*x[24]*x[10]**2*(-sin(x[1])*cos(x[3]) + sin(x[3])*cos(x[1])) + x[14]*x[16]*x[25]*x[10]**2*(-sin(x[1])*cos(x[3]) + sin(x[3])*cos(x[1])) + x[14]*x[17]*x[24]*x[11]**2*(-sin(x[1])*cos(x[4]) + sin(x[4])*cos(x[1])) + x[14]*x[17]*x[25]*x[11]**2*(-sin(x[1])*cos(x[4]) + sin(x[4])*cos(x[1])) + x[14]*x[18]*x[25]*x[12]**2*(-sin(x[1])*cos(x[5]) + sin(x[5])*cos(x[1])), -x[19]*x[15]*x[22]*cos(x[2]) - x[19]*x[15]*x[23]*cos(x[2]) - x[19]*x[15]*x[24]*cos(x[2]) - x[19]*x[15]*x[25]*cos(x[2]) + x[13]*x[15]*x[22]*x[7]**2*(sin(x[0])*cos(x[2]) - sin(x[2])*cos(x[0])) + x[13]*x[15]*x[23]*x[7]**2*(sin(x[0])*cos(x[2]) - sin(x[2])*cos(x[0])) + x[13]*x[15]*x[24]*x[7]**2*(sin(x[0])*cos(x[2]) - sin(x[2])*cos(x[0])) + x[13]*x[15]*x[25]*x[7]**2*(sin(x[0])*cos(x[2]) - sin(x[2])*cos(x[0])) + x[14]*x[15]*x[22]*x[8]**2*(sin(x[1])*cos(x[2]) - sin(x[2])*cos(x[1])) + x[14]*x[15]*x[23]*x[8]**2*(sin(x[1])*cos(x[2]) - sin(x[2])*cos(x[1])) + x[14]*x[15]*x[24]*x[8]**2*(sin(x[1])*cos(x[2]) - sin(x[2])*cos(x[1])) + x[14]*x[15]*x[25]*x[8]**2*(sin(x[1])*cos(x[2]) - sin(x[2])*cos(x[1])) + x[15]*x[16]*x[23]*x[10]**2*(-sin(x[2])*cos(x[3]) + sin(x[3])*cos(x[2])) + x[15]*x[16]*x[24]*x[10]**2*(-sin(x[2])*cos(x[3]) + sin(x[3])*cos(x[2])) + x[15]*x[16]*x[25]*x[10]**2*(-sin(x[2])*cos(x[3]) + sin(x[3])*cos(x[2])) + x[15]*x[17]*x[24]*x[11]**2*(-sin(x[2])*cos(x[4]) + sin(x[4])*cos(x[2])) + x[15]*x[17]*x[25]*x[11]**2*(-sin(x[2])*cos(x[4]) + sin(x[4])*cos(x[2])) + x[15]*x[18]*x[25]*x[12]**2*(-sin(x[2])*cos(x[5]) + sin(x[5])*cos(x[2])), -x[19]*x[16]*x[23]*cos(x[3]) - x[19]*x[16]*x[24]*cos(x[3]) - x[19]*x[16]*x[25]*cos(x[3]) + x[13]*x[16]*x[23]*x[7]**2*(sin(x[0])*cos(x[3]) - sin(x[3])*cos(x[0])) + x[13]*x[16]*x[24]*x[7]**2*(sin(x[0])*cos(x[3]) - sin(x[3])*cos(x[0])) + x[13]*x[16]*x[25]*x[7]**2*(sin(x[0])*cos(x[3]) - sin(x[3])*cos(x[0])) + x[14]*x[16]*x[23]*x[8]**2*(sin(x[1])*cos(x[3]) - sin(x[3])*cos(x[1])) + x[14]*x[16]*x[24]*x[8]**2*(sin(x[1])*cos(x[3]) - sin(x[3])*cos(x[1])) + x[14]*x[16]*x[25]*x[8]**2*(sin(x[1])*cos(x[3]) - sin(x[3])*cos(x[1])) + x[15]*x[16]*x[23]*x[9]**2*(sin(x[2])*cos(x[3]) - sin(x[3])*cos(x[2])) + x[15]*x[16]*x[24]*x[9]**2*(sin(x[2])*cos(x[3]) - sin(x[3])*cos(x[2])) + x[15]*x[16]*x[25]*x[9]**2*(sin(x[2])*cos(x[3]) - sin(x[3])*cos(x[2])) + x[16]*x[17]*x[24]*x[11]**2*(-sin(x[3])*cos(x[4]) + sin(x[4])*cos(x[3])) + x[16]*x[17]*x[25]*x[11]**2*(-sin(x[3])*cos(x[4]) + sin(x[4])*cos(x[3])) + x[16]*x[18]*x[25]*x[12]**2*(-sin(x[3])*cos(x[5]) + sin(x[5])*cos(x[3])), -x[19]*x[17]*x[24]*cos(x[4]) - x[19]*x[17]*x[25]*cos(x[4]) + x[13]*x[17]*x[24]*x[7]**2*(sin(x[0])*cos(x[4]) - sin(x[4])*cos(x[0])) + x[13]*x[17]*x[25]*x[7]**2*(sin(x[0])*cos(x[4]) - sin(x[4])*cos(x[0])) + x[14]*x[17]*x[24]*x[8]**2*(sin(x[1])*cos(x[4]) - sin(x[4])*cos(x[1])) + x[14]*x[17]*x[25]*x[8]**2*(sin(x[1])*cos(x[4]) - sin(x[4])*cos(x[1])) + x[15]*x[17]*x[24]*x[9]**2*(sin(x[2])*cos(x[4]) - sin(x[4])*cos(x[2])) + x[15]*x[17]*x[25]*x[9]**2*(sin(x[2])*cos(x[4]) - sin(x[4])*cos(x[2])) + x[16]*x[17]*x[24]*x[10]**2*(sin(x[3])*cos(x[4]) - sin(x[4])*cos(x[3])) + x[16]*x[17]*x[25]*x[10]**2*(sin(x[3])*cos(x[4]) - sin(x[4])*cos(x[3])) + x[17]*x[18]*x[25]*x[12]**2*(-sin(x[4])*cos(x[5]) + sin(x[5])*cos(x[4])), -x[19]*x[18]*x[25]*cos(x[5]) + x[13]*x[18]*x[25]*x[7]**2*(sin(x[0])*cos(x[5]) - sin(x[5])*cos(x[0])) + x[14]*x[18]*x[25]*x[8]**2*(sin(x[1])*cos(x[5]) - sin(x[5])*cos(x[1])) + x[15]*x[18]*x[25]*x[9]**2*(sin(x[2])*cos(x[5]) - sin(x[5])*cos(x[2])) + x[16]*x[18]*x[25]*x[10]**2*(sin(x[3])*cos(x[5]) - sin(x[5])*cos(x[3])) + x[17]*x[18]*x[25]*x[11]**2*(sin(x[4])*cos(x[5]) - sin(x[5])*cos(x[4]))]
    return out
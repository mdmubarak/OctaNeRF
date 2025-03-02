import taichi as ti
ti.init(arch=ti.gpu)

num_samples = 384

@ti.func
def stratified_sample_ray(ray_origin, ray_direction, t_min, t_max):
    dt = (t_max - t_min) / num_samples
    samples = ti.Vector.field(3, dtype=ti.f32, shape=(num_samples))
    for i in range(num_samples):
        # Using ti.random() to add stratification noise; ensure efficient memory access.
        t = t_min + dt * (i + ti.random())
        samples[i] = ray_origin + ray_direction * t
    return samples

# The above function can be called inside a kernel to populate a field with ray sample positions.

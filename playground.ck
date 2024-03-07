global string ShaderCode;
global int flag;
ShaderTools st;

(
 
st.osc(67)->st.kaleid(4)->st.mask(st.shape(4)->st.rotate())->st.rotate()
->st.blend(st.osc()->st.thresh())->st.repeatX(7)->st.modulate(st.osc())->st.modulateRays(100,800)

).code => ShaderCode;

1 => flag;

// Sources
// osc , voronoi, shape, gradient, noise, solid

// add , diff, mult, sub, blend
// color, colorama, luma, hue, invert, contrast
// kaleid , pixelate, rotate, scale , repeatX, repeatY
// modulate. modulateKaleid, modulateScale, modulateRotate, modulatePixelate
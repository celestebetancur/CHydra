//author: Celeste Betancur Gutierrez

global string ShaderCode;
global int flag;
ShaderTools st;

// Write your CHydra code here and just save the file (ctrl/cmd + s), 
// changes should be visible instantly
(

st.noise(3)->
st.modulatePixelate(st.noise(3)->st.pixelate(80,8),1)->st.shift(0,0,0.5)

).code => ShaderCode;

// chout <= st.shader(ShaderCode);
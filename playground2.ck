10 => GG.camera().posZ;

ShaderTools st;
string ShaderCode;

GCube cube --> GG.scene();
ShaderMaterial cubeMat;
cube.mat(cubeMat);
cube.sca(@(2,2,2));
string cubeCode;

while (true) {
    
    (

    st.voronoi(Math.random2f(2,100))->st.modulate(st.osc(200))

    ).code => ShaderCode;

    (

    st.osc(st.fft[1])

    ).code => cubeCode;

    cube.rotateY(GG.dt());
    cube.rotateZ(GG.dt());
    cube.mat().fragString(st.shader(cubeCode));
    cube.mat().uniformFloat("u_Time", now/second);

    st.back.mat().fragString(st.shader(ShaderCode));
    st.back.mat().uniformFloat("u_Time", now/second);

    GG.nextFrame() => now;
}

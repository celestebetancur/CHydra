/*
Some extra info here:
https://chuck.cs.princeton.edu/doc/program/otfp.html

but for the lazy:
You are going to need two terminals. (as tabs or windows)

In the first one run the main shred:
chuck 2.ck

In the second one you run:
chuck + 2 playground2.ck 

In the second one you can update the shader by running:
chuck ^ 
(this command will give you the VM status, something like: [chuck]: [shred id]: 2 source]: playground2.ck)

then do:
chuck = 2 playground2.ck  (here the number 2 is the ID from the last command)

control + c in the terminal that is running the main shred will kill the chuck
or 
in the second terminal type:
killall chuck 
for a more violent way to do it

author: Celeste Betancur Gutierrez
*/

global string background;
ShaderTools st;

GPlane plane --> GG.scene();
ShaderMaterial shaderMat;
plane.mat(shaderMat);
plane.scaX(16);
plane.scaY(9);

GCube cube --> GG.scene();
ShaderMaterial cubeMat;
cube.mat(cubeMat);
cube.sca(@(2,2,2));
cube.posZ(2);

// This one does not need to be global
string cubeCode;

while (true) {
    
    (

    st.voronoi(Math.sin(now/second))->st.modulate(st.noise(20))->st.color(0.3,0.8,1)

    ).code => background;

    (

    st.osc(60,1,0.6)->st.kaleid()

    ).code => cubeCode;

    cube.rotateY(GG.dt());
    cube.rotateZ(GG.dt());
    cube.mat().fragString(st.shader(cubeCode));
    cube.mat().uniformFloat("u_Time", now/second);

    plane.mat().fragString(st.shader(background));
    plane.mat().uniformFloat("u_Time", now/second);

    GG.nextFrame() => now;
}

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

string background;
ShaderTools st;

GPlane plane --> GG.scene();
ShaderMaterial shaderMat;
plane.mat(shaderMat);
plane.scaX(16);
plane.scaY(9);

GCube cubes[4];
ShaderMaterial cubeMat[4];
// This one does not need to be global
string cubeCode[4];

for(int i; i < cubes.size(); i++){
    cubes[i] --> GG.scene();
    cubes[i].mat(cubeMat[i]);
    cubes[i].sca(@(1,1,1));
}

cubes[0].pos(@(-1.5,-2,3));
cubes[1].pos(@(-0.5,-2,3));
cubes[2].pos(@(0.5,-2,3));
cubes[3].pos(@(1.5,-2,3));

while (true) {
    
    //let's abuse the audio reactivity ;)
    (
    st.solid(0,0,0)
    ).code => background;

    // shader for wrapping cube 1
    (
    st.voronoi(st.FFT[0]*10,st.FOLLOWER)->st.kaleid()->st.diff(st.osc(3,0.1,0.8))
    ).code => cubeCode[0];

    // shader for wrapping cube 2
    (
    st.osc(st.FFT[1]*100,0.5,0.8)
    ).code => cubeCode[1];

    // shader for wrapping cube 3
    (
    st.noise(st.FFT[2]*100)->st.mult(st.osc(10,0.5,0.8))
    ).code => cubeCode[2];

    // shader for wrapping cube 4
    (
    st.gradient(st.FFT[3]*10)
    ).code => cubeCode[3];

    for(int i; i < cubes.size(); i++){
        Math.map(st.FFT[i],0,1,0,6) => float scaler;
        cubes[i].posY(-2+(scaler/2));
        cubes[i].sca(@(1,scaler,1));
        cubes[i].mat().fragString(st.shader(cubeCode[i]));
        cubes[i].mat().uniformFloat("u_Time", now/second);
    }

    plane.mat().fragString(st.shader(background));
    plane.mat().uniformFloat("u_Time", now/second);

    GG.nextFrame() => now;
}

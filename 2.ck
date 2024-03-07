class ShaderType {
    string code;
    string type;
}

public class ShaderTools {     

    // Feature extraction
    adc => FFT fftEx =^ FeatureCollector fc => blackhole;

    fc.upchuck();
    // fc.fvals().size() => int NUM_DIMENSIONS;
    4 => int NUM_DIMENSIONS;

    // set FFT size
    4 => fftEx.size;
    global float fft[NUM_DIMENSIONS];
    // set window type and size
    Windowing.hann(fftEx.size()) => fftEx.window;
    // our hop size (how often to perform analysis)
    (fftEx.size()/2)::samp => dur HOP;

    function void featureExtract(){

        while(true){

            fc.upchuck();

            for( int d; d < NUM_DIMENSIONS; d++ )
                {
                    (fc.fval(d)*100000) => fft[d];
                }
            HOP => now;
        }
    }

    spork~featureExtract();

    GPlane back;
    ShaderMaterial shaderMat;
    back --> GG.scene();
    back.mat(shaderMat);

    back.scaX(32);
    back.scaY(18);
    back.posZ(-10);

    "#version 330 core
    #define PI 3.141592653589793
    #define HALF_PI 1.5707963267948966
    in vec3 v_Pos;
    in vec2 v_TexCoord;
    out vec4 FragColor;
    uniform float u_Time;
    uniform sampler2D tex0;

    float linear(float t) {
        return t;
    }

    float rand(float x, float y){
        return fract(sin(dot(vec2(x, y) ,vec2(12.9898,78.233))) * 43758.5453);
    }

    float sawtooth(float t) {
        return t - floor(t);
    }

    mat2 _rotate(float a) {
        float c = cos(a),
        s = sin(a);
        return mat2(c, -s, s, c);
    }

    vec2 pol2cart(float r, float t) {
        return vec2(r * cos(t), r * sin(t));
    }

    float exponentialIn(float t) {
        return t == 0.0 ? t : pow(2.0, 10.0 * (t - 1.0));
    }

    float exponentialOut(float t) {
        return t == 1.0 ? t : 1.0 - pow(2.0, -10.0 * t);
    }

    float exponentialInOut(float t) {
        return t == 0.0 || t == 1.0
        ? t
        : t < 0.5
        ? +0.5 * pow(2.0, (20.0 * t) - 10.0)
        : -0.5 * pow(2.0, 10.0 - (t * 20.0)) + 1.0;
    }

    float sineIn(float t) {
        return sin((t - 1.0) * HALF_PI) + 1.0;
    }

    float sineOut(float t) {
        return sin(t * HALF_PI);
    }

    float sineInOut(float t) {
        return -0.5 * (cos(PI * t) - 1.0);
    }

    float qinticIn(float t) {
        return pow(t, 5.0);
    }

    float qinticOut(float t) {
        return 1.0 - (pow(t - 1.0, 5.0));
    }

    float qinticInOut(float t) {
        return t < 0.5
        ? +16.0 * pow(t, 5.0)
        : -0.5 * pow(2.0 * t - 2.0, 5.0) + 1.0;
    }

    float quarticIn(float t) {
        return pow(t, 4.0);
    }

    float quarticOut(float t) {
        return pow(t - 1.0, 3.0) * (1.0 - t) + 1.0;
    }

    float quarticInOut(float t) {
        return t < 0.5
        ? +8.0 * pow(t, 4.0)
        : -8.0 * pow(t - 1.0, 4.0) + 1.0;
    }

    float quadraticInOut(float t) {
        float p = 2.0 * t * t;
        return t < 0.5 ? p : -p + (4.0 * t) - 1.0;
    }

    float quadraticIn(float t) {
        return t * t;
    }

    float quadraticOut(float t) {
        return -t * (t - 2.0);
    }

    float cubicIn(float t) {
        return t * t * t;
    }

    float cubicOut(float t) {
        float f = t - 1.0;
        return f * f * f + 1.0;
    }

    float cubicInOut(float t) {
        return t < 0.5
        ? 4.0 * t * t * t
        : 0.5 * pow(2.0 * t - 2.0, 3.0) + 1.0;
    }

    float elasticIn(float t) {
        return sin(13.0 * t * HALF_PI) * pow(2.0, 10.0 * (t - 1.0));
    }

    float elasticOut(float t) {
        return sin(-13.0 * (t + 1.0) * HALF_PI) * pow(2.0, -10.0 * t) + 1.0;
    }

    float elasticInOut(float t) {
        return t < 0.5
        ? 0.5 * sin(+13.0 * HALF_PI * 2.0 * t) * pow(2.0, 10.0 * (2.0 * t - 1.0))
        : 0.5 * sin(-13.0 * HALF_PI * ((2.0 * t - 1.0) + 1.0)) * pow(2.0, -10.0 * (2.0 * t - 1.0)) + 1.0;
    }

    float circularIn(float t) {
        return 1.0 - sqrt(1.0 - t * t);
    }

    float circularOut(float t) {
        return sqrt((2.0 - t) * t);
    }

    float circularInOut(float t) {
        return t < 0.5
        ? 0.5 * (1.0 - sqrt(1.0 - 4.0 * t * t))
        : 0.5 * (sqrt((3.0 - 2.0 * t) * (2.0 * t - 1.0)) + 1.0);
    }

    float bounceOut(float t) {
        const float a = 4.0 / 11.0;
        const float b = 8.0 / 11.0;
        const float c = 9.0 / 10.0;
        
        const float ca = 4356.0 / 361.0;
        const float cb = 35442.0 / 1805.0;
        const float cc = 16061.0 / 1805.0;
        
        float t2 = t * t;
        
        return t < a
        ? 7.5625 * t2
        : t < b
        ? 9.075 * t2 - 9.9 * t + 3.4
        : t < c
        ? ca * t2 - cb * t + cc
        : 10.8 * t * t - 20.52 * t + 10.72;
    }

    float bounceIn(float t) {
        return 1.0 - bounceOut(1.0 - t);
    }

    float bounceInOut(float t) {
        return t < 0.5
        ? 0.5 * (1.0 - bounceOut(1.0 - t * 2.0))
        : 0.5 * bounceOut(t * 2.0 - 1.0) + 0.5;
    }

    float backIn(float t) {
        return pow(t, 3.0) - t * sin(t * PI);
    }

    float backOut(float t) {
        float f = 1.0 - t;
        return 1.0 - (pow(f, 3.0) - f * sin(f * PI));
    }

    float backInOut(float t) {
        float f = t < 0.5
        ? 2.0 * t
        : 1.0 - (2.0 * t - 1.0);
        
        float g = pow(f, 3.0) - f * sin(f * PI);
        
        return t < 0.5
        ? 0.5 * g
        : 0.5 * (1.0 - g) + 0.5;
    }

    //-------------------------------------------------------------

    vec3 palette( float t ) {
        vec3 a = vec3(0.5, 0.5, 0.5);
        vec3 b = vec3(0.5, 0.5, 0.5);
        vec3 c = vec3(1.0, 1.0, 1.0);
        vec3 d = vec3(0.263,0.416,0.557);

        return a + b*cos( 6.28318*(c*t+d) );
    }

    vec4 permute(vec4 x){
        return mod(((x*34.0)+1.0)*x, 289.0);
    }

    vec4 taylorInvSqrt(vec4 r){
        return 1.79284291400159 - 0.85373472095314 * r;
    }

    vec2 fade(vec2 t) {
        return t*t*t*(t*(t*6.0-15.0)+10.0);
    }

    float _noise(vec3 v){
        const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
        const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);
        // First corner
        vec3 i  = floor(v + dot(v, C.yyy) );
        vec3 x0 =   v - i + dot(i, C.xxx) ;
        // Other corners
        vec3 g = step(x0.yzx, x0.xyz);
        vec3 l = 1.0 - g;
        vec3 i1 = min( g.xyz, l.zxy );
        vec3 i2 = max( g.xyz, l.zxy );
        //  x0 = x0 - 0. + 0.0 * C
        vec3 x1 = x0 - i1 + 1.0 * C.xxx;
        vec3 x2 = x0 - i2 + 2.0 * C.xxx;
        vec3 x3 = x0 - 1. + 3.0 * C.xxx;
        // Permutations
        i = mod(i, 289.0 );
        vec4 p = permute( permute( permute(
                                        i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
                                + i.y + vec4(0.0, i1.y, i2.y, 1.0 ))
                        + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));
        // Gradients
        // ( N*N points uniformly over a square, mapped onto an octahedron.)
        float n_ = 1.0/7.0; // N=7
        vec3  ns = n_ * D.wyz - D.xzx;
        vec4 j = p - 49.0 * floor(p * ns.z *ns.z);  //  mod(p,N*N)
        vec4 x_ = floor(j * ns.z);
        vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)
        vec4 x = x_ *ns.x + ns.yyyy;
        vec4 y = y_ *ns.x + ns.yyyy;
        vec4 h = 1.0 - abs(x) - abs(y);
        vec4 b0 = vec4( x.xy, y.xy );
        vec4 b1 = vec4( x.zw, y.zw );
        vec4 s0 = floor(b0)*2.0 + 1.0;
        vec4 s1 = floor(b1)*2.0 + 1.0;
        vec4 sh = -step(h, vec4(0.0));
        vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
        vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;
        vec3 p0 = vec3(a0.xy,h.x);
        vec3 p1 = vec3(a0.zw,h.y);
        vec3 p2 = vec3(a1.xy,h.z);
        vec3 p3 = vec3(a1.zw,h.w);
        //Normalise gradients
        vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
        p0 *= norm.x;
        p1 *= norm.y;
        p2 *= norm.z;
        p3 *= norm.w;
        // Mix final noise value
        vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
        m = m * m;
        return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1),
                                    dot(p2,x2), dot(p3,x3) ) );
    }

    vec3 _rgbToHsv(vec3 c){
        vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
        vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
        vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
        float d = q.x - min(q.w, q.y);
        float e = 1.0e-10;
        return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
    }

    vec3 _hsvToRgb(vec3 c){
        vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
        vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
        return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
    }

    float _luminance(vec3 rgb){
        const vec3 W = vec3(0.2125, 0.7154, 0.0721);
        return dot(rgb, W);
    }

    float map(float value, float inputMin, float inputMax, float outputMin, float outputMax, bool clamp){
        if (abs(inputMin - inputMax) < 0.000000001) {
            return outputMin;
        }
        else {
            float outVal = ((value - inputMin) / (inputMax - inputMin) * (outputMax - outputMin) + outputMin);
                        
            if (clamp) {
                if (outputMax < outputMin) {
                    if (outVal < outputMax)outVal = outputMax;
                    else if (outVal > outputMin)outVal = outputMin;
                }
            else {
                if (outVal > outputMax)outVal = outputMax;
                else if (outVal < outputMin)outVal = outputMin;
                }
            }
            return outVal;
        }
    }

    vec4 noise(vec2 _st, float scale, float offset){
        return vec4(vec3(_noise(vec3(_st*scale, offset*u_Time))), 1.0);
    }

    vec4 voronoi(vec2 _st, float scale, float speed, float blending){
        vec3 color = vec3(0.0);
    // Scale
        _st *= scale;
    // Tile the space
        vec2 i_st = floor(_st);
        vec2 f_st = fract(_st);
        float m_dist = 10.0;  // minimun distance
        vec2 m_point;        // minimum point
        for (int j=-1; j<=1; j++ ) {
            for (int i=-1; i<=1; i++ ) {
                vec2 neighbor = vec2(float(i),float(j));
                vec2 p = i_st + neighbor;
                vec2 point = fract(sin(vec2(dot(p,vec2(127.1,311.7)),dot(p,vec2(269.5,183.3))))*43758.5453);
                point = 0.5 + 0.5*sin(u_Time*speed + 6.2831*point);
                vec2 diff = neighbor + point - f_st;
                float dist = length(diff);
                if( dist < m_dist ) {
                    m_dist = dist;
                    m_point = point;
                }
            }
        }
    // Assign a color using the closest point position
    color += dot(m_point,vec2(.3,.6));
    color *= 1.0 - blending*m_dist;
    return vec4(color, 1.0);
    }

    vec4 src(vec2 _st, sampler2D tex){
        return texture(tex,_st);
    }

    vec4 osc(vec2 _st, float freq, float sync, float offset){
        vec2 st = _st;
        float r = sin((st.x-offset*2.0/freq+u_Time*sync)*freq)*0.5  + 0.5;
        float g = sin((st.x+u_Time*sync)*freq)*0.5 + 0.5;
        float b = sin((st.x+offset/freq+u_Time*sync)*freq)*0.5  + 0.5;
        return vec4(r, g, b, 1.0);
    }

    vec4 shape(vec2 _st, float sides, float radius, float smoothing){
        vec2 st = _st * 2. - 1.;
        float a = atan(st.x,st.y) + PI;
        float r = (2.* PI)/sides;
        float d = cos(floor(.5+a/r)*r-a)*length(st);
        return vec4(vec3(1.0-smoothstep(radius,radius + smoothing,d)), 1.0);
    }

    vec4 gradient(vec2 _st, float speed){
        return vec4(_st, sin(u_Time*speed), 1.0);
    }

    vec4 solid (float r, float g, float b, float a){
        return vec4(r, g, b, a);
    }

    vec2 rotate(vec2 _st, float angle, float speed){
        vec2 xy = _st - vec2(0.5);
        float ang = angle + speed * u_Time;
        xy = mat2(cos(ang),-sin(ang), sin(ang),cos(ang))*xy;
        xy += 0.5;
        return xy;
    }

    vec2 scale(vec2 _st, float amount, float xMult, float yMult, float offsetX, float offsetY){
        vec2 xy = _st - vec2(offsetX, offsetY);
        xy*=(1.0/vec2(amount*xMult, amount*yMult));
        xy+=vec2(offsetX, offsetY);
        return xy;
    }

    vec2 pixelate (vec2 _st, float pixelX, float pixelY){
        vec2 xy = vec2(pixelX, pixelY);
        return (floor(_st * xy) + 0.5)/xy;
    }

    vec2 repeat(vec2 _st, float repeatX, float repeatY, float offsetX, float offsetY){
        vec2 st = _st * vec2(repeatX, repeatY);
        st.x += step(1., mod(st.y,2.0)) * offsetX;
        st.y += step(1., mod(st.x,2.0)) * offsetY;
        return fract(st);
    }

    vec2 modulateRepeat(vec2 _st, vec4 _c0, float repeatX, float repeatY, float offsetX, float offsetY){
        vec2 st = _st * vec2(repeatX, repeatY);
        st.x += step(1., mod(st.y,2.0)) + _c0.r * offsetX;
        st.y += step(1., mod(st.x,2.0)) + _c0.g * offsetY;
        return fract(st);
    }

    vec2 repeatX (vec2 _st, float reps, float offset){
        vec2 st = _st * vec2(1.0, reps);
        //  float f =  mod(_st.y,2.0);
        st.x += step(1., mod(st.x,2.0))* offset;
        return fract(st);
    }

    vec2 modulateRepeatX(vec2 _st, vec4 _c0, float reps, float offset){
        vec2 st = _st * vec2(reps,1.0);
        st.y += step(1.0, mod(st.x,2.0)) + _c0.r * offset;
        return fract(st);
    }

    vec2 repeatY (vec2 _st, float reps, float offset){
        vec2 st = _st * vec2(reps, 1.0);
        //  float f =  mod(_st.y,2.0);
        st.y += step(1., mod(st.y,2.0))* offset;
        return fract(st);
    }

    vec2 modulateRepeatY(vec2 _st, vec4 _c0, float reps, float offset){
        vec2 st = _st * vec2(reps,1.0);
        st.x += step(1.0, mod(st.y,2.0)) + _c0.r * offset;
        return fract(st);
    }

    vec2 kaleid(vec2 _st, float nSides){
        vec2 st = _st;
        st -= 0.5;
        float r = length(st);
        float a = atan(st.y, st.x);
        float pi = 2.* PI;
        a = mod(a,pi/nSides);
        a = abs(a-pi/nSides/2.);
        return r*vec2(cos(a), sin(a));
    }

    vec2 modulateKaleid(vec2 _st, vec4 _c0, float nSides){
        vec2 st = _st - 0.5;
        float r = length(st);
        float a = atan(st.y, st.x);
        float pi = 2.* PI;
        a = mod(a,pi/nSides);
        a = abs(a-pi/nSides/2.);
        return (_c0.r+r)*vec2(cos(a), sin(a));
    }


    vec2 scroll(vec2 _st, float scrollX, float scrollY, float speedX, float speedY){
        _st.x += scrollX + u_Time*speedX;
        _st.y += scrollY + u_Time*speedY;
        return _st;
    }

    vec2 scrollX (vec2 _st, float scrollX, float speed){
        _st.x += scrollX + u_Time*speed;
        return _st;
    }

    vec2 modulateScrollX (vec2 _st, vec4 _c0, float scrollX, float speed){
        _st.x += _c0.r*scrollX + u_Time*speed;
    return fract(_st);
    }

    vec2 scrollY (vec2 _st, float scrollY, float speed){
        _st.y += scrollY + u_Time*speed;
        return _st;
    }

    vec2 modulateScrollY (vec2 _st, vec4 _c0, float scrollY, float speed){
        _st.y += _c0.r*scrollY + u_Time*speed;
    return fract(_st);
    }

    vec4 posterize(vec4 _c0, float bins, float gamma){
        vec4 c2 = pow(_c0, vec4(gamma));
        c2 *= vec4(bins);
        c2 = floor(c2);
        c2/= vec4(bins);
        c2 = pow(c2, vec4(1.0/gamma));
        return vec4(c2.xyz, _c0.a);
    }

    vec4 shift(vec4 _c0, float r, float g, float b, float a){
        vec4 c2 = vec4(_c0);
        c2.r = fract(c2.r + r);
        c2.g = fract(c2.g + g);
        c2.b = fract(c2.b + b);
        c2.a = fract(c2.a + a);
        return vec4(c2.rgba);
    }

    vec4 add(vec4 _c0, vec4 _c1, float amount){
        return (_c0+_c1)*amount + _c0*(1.0-amount);
    }

    vec4 sub(vec4 _c0, vec4 _c1, float amount){
        return (_c0-_c1)*amount + _c0*(1.0-amount);
    }
    vec4 layer(vec4 _c0, vec4 _c1){
        return vec4(mix(_c0.rgb, _c1.rgb, _c1.a), _c0.a+_c1.a);
    }

    vec4 blend(vec4 _c0, vec4 _c1, float amount){
        return _c0*(1.0-amount)+_c1*amount;
    }

    vec4 mult(vec4 _c0, vec4 _c1, float amount){
        return _c0*(1.0-amount)+(_c0*_c1)*amount;
    }

    vec4 diff(vec4 _c0, vec4 _c1, float amount){
        return vec4(abs(_c0.rgb-_c1.rgb), max(_c0.a, _c1.a));
    }

    vec2 modulate(vec2 _st, vec4 _c0, float amount){
        return _st + _c0.xy * amount;
    }

    vec2 modulateScale(vec2 _st, vec4 _c0, float offset, float multiple){
        vec2 xy = _st - vec2(0.5);
        xy*=(1.0/vec2(offset + multiple*_c0.r, offset + multiple*_c0.g));
        xy+=vec2(0.5);
        return xy;
    }

    vec2 modulatePixelate(vec2 _st, vec4 _c0, float offset, float multiple){
        vec2 xy = vec2(offset + _c0.x*multiple, offset + _c0.y*multiple);
        return (floor(_st * xy) + 0.5)/xy;
    }

    vec2 modulateRotate(vec2 _st, vec4 _c0, float offset, float multiple){
        vec2 xy = _st - vec2(0.5);
        float angle = offset + _c0.x * multiple;
        xy = mat2(cos(angle),-sin(angle), sin(angle),cos(angle))*xy;
        xy += 0.5;
        return xy;
    }

    vec2 modulateHue(vec2 _st,vec4 _c0, float amount){
        vec2 t = _st + vec2(_c0.g - _c0.r, _c0.b - _c0.g) * amount / v_TexCoord;
        return t;
    }

    vec4 invert(vec4 _c0, float amount){
        return vec4((1.0-_c0.rgb)*amount + _c0.rgb*(1.0-amount), _c0.a);
    }

    vec4 contrast(vec4 _c0, float amount){
        vec4 c = (_c0-vec4(0.5))*vec4(amount) + vec4(0.5);
        return vec4(c.rgb, _c0.a);
    }

    vec4 brightness(vec4 _c0, float amount){
        return vec4(_c0.rgb + vec3(amount), _c0.a);
    }

    vec4 mask(vec4 _c0, vec4 _c1){
        float a = _luminance(_c1.rgb);
        return vec4(_c0.rgb*a, a*_c0.a);
    }

    vec4 luma(vec4 _c0, float threshold, float tolerance){
        float a = smoothstep(threshold-(tolerance+0.0000001), threshold+(tolerance+0.0000001), _luminance(_c0.rgb));
        return vec4(_c0.rgb*a, a);
    }

    vec4 thresh(vec4 _c0, float threshold, float tolerance){
        return vec4(vec3(smoothstep(threshold-tolerance, threshold+tolerance, _luminance(_c0.rgb))), _c0.a);
    }

    vec4 color(vec4 _c0, float r, float g, float b, float a){
        vec4 c = vec4(r, g, b, a);
        vec4 pos = step(0.0, c); // detect whether negative
        // if > 0, return r * _c0
        // if < 0 return (1.0-r) * _c0
        return vec4(mix((1.0-_c0)*abs(c), c*_c0, pos));
    }

    vec4 r(vec4 _c0, float scale, float offset){
        return vec4(_c0.r * scale + offset);
    }

    vec4 g(vec4 _c0, float scale, float offset){
        return vec4(_c0.g * scale + offset);
    }

    vec4 b(vec4 _c0, float scale, float offset){
        return vec4(_c0.b * scale + offset);
    }

    vec4 a(vec4 _c0, float scale, float offset){
        return vec4(_c0.a * scale + offset);
    }

    vec4 saturate(vec4 _c0, float amount){
        const vec3 W = vec3(0.2125,0.7154,0.0721);
        vec3 intensity = vec3(dot(_c0.rgb,W));
        return vec4(mix(intensity,_c0.rgb,amount), _c0.a);
    }

    vec4 hue(vec4 _c0, float hue){
        vec3 c = _rgbToHsv(_c0.rgb);
        c.r += hue;
    //  c.r = fract(c.r);
        return vec4(_hsvToRgb(c), _c0.a);
    }

    vec4 colorama(vec4 _c0, float amount){
        vec3 c = _rgbToHsv(_c0.rgb);
        c += vec3(amount);
        c = _hsvToRgb(c);
        c = fract(c);
        return vec4(c, _c0.a);
    }

    vec2 modulateSR(vec2 _st, vec4 _c0, float multiple, float offset, float rotateMultiple, float rotateOffset){
        vec2 xy = _st - vec2(0.5);
        float angle = rotateOffset + _c0.z * rotateMultiple;
        xy = mat2(cos(angle),-sin(angle), sin(angle),cos(angle))*xy;
        xy *= (1.0/vec2(offset + multiple*_c0.r, offset + multiple*_c0.g));
        xy += vec2(0.5);
        return xy;
    }

    vec4 chroma(vec4 _c0){
        float maxrb = max( _c0.r, _c0.b );
        float k = clamp( (_c0.g-maxrb)*5.0, 0.0, 1.0 );
        float dg = _c0.g; 
        _c0.g = min( _c0.g, maxrb*0.8 ); 
        _c0 += vec4(dg - _c0.g);
        return vec4(_c0.rgb, 1.0 - k);
    }

    vec2 sphere(vec2 _st, float radius, float rot){
        vec2 pos = _st-0.5;
        vec3 rpos = vec3(0.0, 0.0, -10.0);
        vec3 rdir = normalize(vec3(pos * 3.0, 1.0));
        float d = 0.0;
        for(int i = 0; i < 16; ++i){
            d = length(rpos) - radius;
            rpos += d * rdir;
            if (abs(d) < 0.001)break;
        }
        return vec2(atan(rpos.z, rpos.x)+rot, atan(length(rpos.xz), rpos.y));
    }

    vec2 sphereDisplacement(vec2 _st, vec4 _c0, float radius, float rot){
        vec2 pos = _st-0.5;
        vec3 rpos = vec3(0.0, 0.0, -10.0);
        vec3 rdir = normalize(vec3(pos * 3.0, 1.0));
        float d = 0.0;
        for(int i = 0; i < 16; ++i){
            float height = length(_c0);
            d = length(rpos) - (radius+height);
            rpos += d * rdir;
            if (abs(d) < 0.001)break;
        }
        return vec2(atan(rpos.z, rpos.x)+rot, atan(length(rpos.xz), rpos.y));
    }

    vec4 fractal(vec2 _st) {

        vec2 uv = _st;
        vec2 uv0 = uv;

        vec3 finalColor = vec3(0.0);
        
        for (float i = 0.0; i < 4.0; i++) {
            uv = fract(uv * 1.5) - 0.5;

            float d = length(uv) * exp(-length(uv0));

            vec3 col = palette(length(uv0) + i*.4 + u_Time*.4);

            d = sin(d*8. + u_Time)/8.;
            d = abs(d);

            d = pow(0.01 / d, 1.2);

            finalColor += col * d;
        }
            
        return vec4(finalColor, 1.0);
    }

    vec4 modulateRays( vec2 _st , vec4 _c0, float scale, float samples)
    {
        vec2 uv = _st;
        
        vec3 col = vec3(0.0);

        for (int i = 0; i < floor(samples); i++) {
            scale -= 0.0002;
            uv -= 0.5;
            uv *= scale;
            uv += 0.5;
            col += smoothstep(0.0, 1.0, _c0.rgb * 0.08);
        }

        return vec4(col,1.0);
    }

    void main(){
        vec2 uv = v_TexCoord;
        FragColor = " => string header;

    ";}" => string close;

    function ShaderType noise(){
        ShaderType toReturn;
        "noise ( uv , 100. , 10. )"=> toReturn.code;
        "src" => toReturn.type;
        return toReturn;
    }

    function ShaderType noise(float freq){
        ShaderType toReturn;
        "noise ( uv , "+Std.ftoa(freq,10)+" , 10. )"=> toReturn.code;
        "src" => toReturn.type;
        return toReturn;
    }

    function ShaderType noise(float freq, float offset){
        ShaderType toReturn;
        "noise ( uv , "+Std.ftoa(freq,10)+" , "+Std.ftoa(offset,10)+" )"=> toReturn.code;
        "src" => toReturn.type;
        return toReturn;
    }

    function ShaderType voronoi(){
        ShaderType toReturn;
        "voronoi ( uv , 100. , 1.0 , 0.3 )"=> toReturn.code;
        "src" => toReturn.type;
        return toReturn;
    } // scale speed blending

    function ShaderType voronoi(float scale){
        ShaderType toReturn;
        "voronoi ( uv , "+Std.ftoa(scale,10)+" , 1.0 , 0.3 )"=> toReturn.code;
        "src" => toReturn.type;
        return toReturn;
    }

    function ShaderType voronoi(float scale, float speed){
        ShaderType toReturn;
        "voronoi ( uv , "+Std.ftoa(scale,10)+" , "+Std.ftoa(speed,10)+" , 0.3 )"=> toReturn.code;
        "src" => toReturn.type;
        return toReturn;
    }

    function ShaderType voronoi(float scale, float speed, float blending){
        ShaderType toReturn;
        "voronoi ( uv , "+Std.ftoa(scale,10)+" , "+Std.ftoa(speed,10)+" , "+Std.ftoa(blending,10)+" )"=> toReturn.code;
        "src" => toReturn.type;
        return toReturn;
    }
    //TODO: think about this
    function ShaderType src(){
        ShaderType toReturn;
        "src ( uv , tex0)"=> toReturn.code;
        "src" => toReturn.type;
        return toReturn;
    }

    function ShaderType osc(){
        ShaderType toReturn;
        "osc ( uv , 100. , 0.1 , 0.0 )"=> toReturn.code;
        "src" => toReturn.type;
        return toReturn;
    }

    function ShaderType osc(float freq){
        ShaderType toReturn;
        "osc ( uv , "+Std.ftoa(freq,10)+" , 0.1 , 0.0 )"=> toReturn.code;
        "src" => toReturn.type;
        return toReturn;
    }
    
    function ShaderType osc(float freq, float sync){
        ShaderType toReturn;
        "osc ( uv , "+Std.ftoa(freq,10)+" , "+Std.ftoa(sync,10)+",0.0 )"=> toReturn.code;
        "src" => toReturn.type;
        return toReturn;
    }

    function ShaderType osc(float freq, float sync, float offset){
        ShaderType toReturn;
        "osc ( uv , "+Std.ftoa(freq,10)+" , "+Std.ftoa(sync,10)+" , "+Std.ftoa(offset,10)+" )"=> toReturn.code;
        "src" => toReturn.type;
        return toReturn;
    }

    function ShaderType shape(){
        ShaderType toReturn;
        "shape ( uv , 4. , 0.25 , 0.0 )"=> toReturn.code;
        "src" => toReturn.type;
        return toReturn;
    }

    function ShaderType shape(float sides){
        ShaderType toReturn;
        "shape ( uv , "+Std.ftoa(sides,10)+" , 0.25 , 0.0 )"=> toReturn.code;
        "src" => toReturn.type;
        return toReturn;
    }

    function ShaderType shape(float sides, float radius){
        ShaderType toReturn;
        "shape ( uv , "+Std.ftoa(sides,10)+" , "+Std.ftoa(radius,10)+" , 0.0 )"=> toReturn.code;
        "src" => toReturn.type;
        return toReturn;
    }

    function ShaderType shape(float sides, float radius, float smoothing){
        ShaderType toReturn;
        "shape ( uv , "+Std.ftoa(sides,10)+" , "+Std.ftoa(radius,10)+" , "+Std.ftoa(smoothing,10)+" )"=> toReturn.code;
        "src" => toReturn.type;
        return toReturn;
    }

    function ShaderType gradient(){
        ShaderType toReturn;
        "gradient ( uv , 2.0 )"=> toReturn.code;
        "src" => toReturn.type;
        return toReturn;
    }

    function ShaderType gradient(float speed){
        ShaderType toReturn;
        "gradient ( uv , "+Std.ftoa(speed,10)+" )"=> toReturn.code;
        "src" => toReturn.type;
        return toReturn;
    }

    function ShaderType solid(){
        ShaderType toReturn;
        "solid ( 0.0, 0.0, 0.0, 1.0)"=> toReturn.code;
        "src" => toReturn.type;
        return toReturn;
    }

    function ShaderType solid(float r){
        ShaderType toReturn;
        "solid ( "+Std.ftoa(r,10)+", 0.0, 0.0, 1.0)"=> toReturn.code;
        "src" => toReturn.type;
        return toReturn;
    }

    function ShaderType solid(float r, float g){
        ShaderType toReturn;
        "solid ( "+Std.ftoa(r,10)+", "+Std.ftoa(g,10)+", 0.0, 1.0)"=> toReturn.code;
        "src" => toReturn.type;
        return toReturn;
    }

    function ShaderType solid(float r, float g, float b){
        ShaderType toReturn;
        "solid ( "+Std.ftoa(r,10)+", "+Std.ftoa(g,10)+", "+Std.ftoa(b,10)+", 1.0)"=> toReturn.code;
        "src" => toReturn.type;
        return toReturn;
    }

    function ShaderType solid(float r, float g, float b, float a){
        ShaderType toReturn;
        "solid ( "+Std.ftoa(r,10)+", "+Std.ftoa(g,10)+", "+Std.ftoa(b,10)+", "+Std.ftoa(a,10)+")"=> toReturn.code;
        "src" => toReturn.type;
        return toReturn;
    }

    function ShaderType rotate(){
        ShaderType toReturn;
        "rotate ( uv , 1. , 1. )"=> toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType rotate(float angle){
        ShaderType toReturn;
        "rotate ( uv , "+Std.ftoa(angle,10)+" , 1. )"=> toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType rotate(float angle, float speed){
        ShaderType toReturn;
        "rotate ( uv , "+Std.ftoa(angle,10)+" , "+Std.ftoa(speed,10)+" )"=> toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType scale(){
        ShaderType toReturn;
        "scale ( uv , 1.5 , 1.0 , 1.0 , 0.5 , 0.5)"=> toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType scale(float amount){
        ShaderType toReturn;
        "scale ( uv , "+Std.ftoa(amount,10)+" , 1.0 , 1.0 , 0.5 , 0.5)"=> toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType scale(float amount, float xMult){
        ShaderType toReturn;
        "scale ( uv , "+Std.ftoa(amount,10)+" , "+Std.ftoa(xMult,10)+" , 1.0 , 0.5 , 0.5)"=> toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType scale(float amount, float xMult, float yMult){
        ShaderType toReturn;
        "scale ( uv , "+Std.ftoa(amount,10)+" , "+Std.ftoa(xMult,10)+" , "+Std.ftoa(yMult,10)+" , 0.5 , 0.5)"=> toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType scale(float amount, float xMult, float yMult, float offsetX){
        ShaderType toReturn;
        "scale ( uv , "+Std.ftoa(amount,10)+" , "+Std.ftoa(xMult,10)+" , "+Std.ftoa(yMult,10)+" , "+Std.ftoa(offsetX,10)+" , 0.5 )"=> toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType scale(float amount, float xMult, float yMult, float offsetX, float offsetY){
        ShaderType toReturn;
        "scale ( uv , "+Std.ftoa(amount,10)+" , "+Std.ftoa(xMult,10)+" , "+Std.ftoa(yMult,10)+" , "+Std.ftoa(offsetX,10)+" , "+Std.ftoa(offsetY,10)+" )"=> toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType pixelate(){
        ShaderType toReturn;
        "pixelate ( uv , 8. , 8. )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType pixelate(float pixelX){
        ShaderType toReturn;
        "pixelate ( uv , "+Std.ftoa(pixelX,10)+" , 8. )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType pixelate(float pixelX, float pixelY){
        ShaderType toReturn;
        "pixelate ( uv , "+Std.ftoa(pixelX,10)+" , "+Std.ftoa(pixelX,10)+" )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType repeatTex(){
        ShaderType toReturn;
        "repeat ( uv , 3. , 3. , 0. , 0. )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType repeatTex(float repeatX){
        ShaderType toReturn;
        "repeat ( uv , "+Std.ftoa(repeatX,10)+" , 3. , 0. , 0. )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType repeatTex(float repeatX, float repeatY){
        ShaderType toReturn;
        "repeat ( uv , "+Std.ftoa(repeatX,10)+" , "+Std.ftoa(repeatY,10)+" , 0. , 0. )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType repeatTex(float repeatX, float repeatY, float offsetX){
        ShaderType toReturn;
        "repeat ( uv , "+Std.ftoa(repeatX,10)+" , "+Std.ftoa(repeatY,10)+" , "+Std.ftoa(offsetX,10)+" , 0. )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType repeatTex(float repeatX, float repeatY, float offsetX, float offsetY){
        ShaderType toReturn;
        "repeat ( uv , "+Std.ftoa(repeatX,10)+" , "+Std.ftoa(repeatY,10)+" , "+Std.ftoa(offsetX,10)+" , "+Std.ftoa(offsetY,10)+" )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

        function ShaderType modulateRepeat(ShaderType src){
        ShaderType toReturn;
        "modulateRepeat ( tx , "+src.code+" , 3. , 3. , 0.5 , 0.5 )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateRepeat(ShaderType src, float repeatX){
        ShaderType toReturn;
        "modulateRepeat ( tx , "+src.code+" , "+Std.ftoa(repeatX,10)+" , 3. , 0.5 , 0.5 )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateRepeat(ShaderType src, float repeatX, float repeatY){
        ShaderType toReturn;
        "modulateRepeat ( tx , "+src.code+" , "+Std.ftoa(repeatX,10)+" , "+Std.ftoa(repeatY,10)+" , 0.5 , 0.5 )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateRepeat(ShaderType src, float repeatX, float repeatY, float offsetX){
        ShaderType toReturn;
        "modulateRepeat ( tx , "+src.code+" , "+Std.ftoa(repeatX,10)+" , "+Std.ftoa(repeatY,10)+" , "+Std.ftoa(offsetX,10)+" , 0.5 )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateRepeat(ShaderType src, float repeatX, float repeatY, float offsetX, float offsetY){
        ShaderType toReturn;
        "modulateRepeat ( tx , "+src.code+" , "+Std.ftoa(repeatX,10)+" , "+Std.ftoa(repeatY,10)+" , "+Std.ftoa(offsetX,10)+" , "+Std.ftoa(offsetX,10)+" )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType repeatX(){
        ShaderType toReturn;
        "repeatX ( uv , 4. , 4. )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType repeatX(float reps){
        ShaderType toReturn;
        "repeatX ( uv , "+Std.ftoa(reps,10)+" , 4. )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType repeatX(float reps, float offset){
        ShaderType toReturn;
        "repeatX ( uv , "+Std.ftoa(reps,10)+" , "+Std.ftoa(offset,10)+" )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateRepeatX(ShaderType src){
        ShaderType toReturn;
        "modulateRepeatX ( tx , "+src.code+" , 3. , 0.5 )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateRepeatX(ShaderType src, float reps){
        ShaderType toReturn;
        "modulateRepeatX ( tx , "+src.code+" , "+Std.ftoa(reps,10)+" , 0.5 )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateRepeatX(ShaderType src, float reps, float offset){
        ShaderType toReturn;
        "modulateRepeatX ( tx , "+src.code+" , "+Std.ftoa(reps,10)+" , "+Std.ftoa(reps,10)+" )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType repeatY(){
        ShaderType toReturn;
        "repeatY ( uv , 3. , 0.5 )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType repeatY(float reps){
        ShaderType toReturn;
        "repeatY ( uv , "+Std.ftoa(reps,10)+" , .5 )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType repeatY(float reps, float offset){
        ShaderType toReturn;
        "repeatY ( uv , "+Std.ftoa(reps,10)+" , "+Std.ftoa(offset,10)+" )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateRepeatY(ShaderType src){
        ShaderType toReturn;
        "modulateRepeatY ( tx , "+src.code+" , 3. , 0.5 )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateRepeatY(ShaderType src, float reps){
        ShaderType toReturn;
        "modulateRepeatY ( tx , "+src.code+" , "+Std.ftoa(reps,10)+" , 0.5 )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateRepeatY(ShaderType src, float reps, float offset){
        ShaderType toReturn;
        "modulateRepeatY ( tx , "+src.code+" , "+Std.ftoa(reps,10)+" , "+Std.ftoa(reps,10)+" )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType kaleid(){
        ShaderType toReturn;
        "kaleid ( uv , 4. )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType kaleid(float nSides){
        ShaderType toReturn;
        "kaleid ( uv , "+Std.ftoa(nSides,10)+" )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateKaleid(ShaderType src){
        ShaderType toReturn;
        "modulateKaleid ( uv , "+src.code+" , 4. )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateKaleid(ShaderType src, float nSides){
        ShaderType toReturn;
        "modulateKaleid ( uv , "+src.code+" , "+Std.ftoa(nSides,10)+" )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType scroll(){
        ShaderType toReturn;
        "scroll ( uv , 5. , 5. , 0.5 , 0.5 )"=> toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType scroll(float scrollX){
        ShaderType toReturn;
        "scroll ( uv , "+Std.ftoa(scrollX,10)+" , 5. , 0.5 , 0.5 )"=> toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType scroll(float scrollX, float scrollY){
        ShaderType toReturn;
        "scroll ( uv , "+Std.ftoa(scrollX,10)+" , "+Std.ftoa(scrollY,10)+" , 0.5 , 0.5 )"=> toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType scroll(float scrollX, float scrollY, float speedX){
        ShaderType toReturn;
        "scroll ( uv , "+Std.ftoa(scrollX,10)+" , "+Std.ftoa(scrollY,10)+" , "+Std.ftoa(speedX,10)+" , 0.5 )"=> toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType scroll(float scrollX, float scrollY, float speedX, float speedY){
        ShaderType toReturn;
        "scroll ( uv , "+Std.ftoa(scrollX,10)+" , "+Std.ftoa(scrollY,10)+" , "+Std.ftoa(speedX,10)+" , "+Std.ftoa(speedY,10)+" )"=> toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType scrollX(){
        ShaderType toReturn;
        "scrollX ( uv , 5. , 0.5 )"=> toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType scrollX(float scrollX){
        ShaderType toReturn;
        "scrollX ( uv , "+Std.ftoa(scrollX,10)+" , 0.5 )"=> toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType scrollX(float scrollX, float speed){
        ShaderType toReturn;
        "scrollX ( uv , "+Std.ftoa(scrollX,10)+" , "+Std.ftoa(speed,10)+" )"=> toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType scrollY(){
        ShaderType toReturn;
        "scrollY ( uv , 5. , 0.5 )"=> toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType scrollY(float scrollY){
        ShaderType toReturn;
        "scrollY ( uv , "+Std.ftoa(scrollY,10)+" , 0.5 )"=> toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType scrollY(float scrollY, float speed){
        ShaderType toReturn;
        "scrollY ( uv , "+Std.ftoa(scrollY,10)+" , "+Std.ftoa(speed,10)+" )"=> toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateScrollX(ShaderType src){
        ShaderType toReturn;
        "modulateScrollX ( tx , "+src.code+" , 0.5 , 1.0 )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateScrollX(ShaderType src, float scrollX){
        ShaderType toReturn;
        "modulateScrollX ( tx , "+src.code+" , "+Std.ftoa(scrollX,10)+" , 1.0 )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateScrollX(ShaderType src, float scrollX, float speed){
        ShaderType toReturn;
        "modulateScrollX ( tx , "+src.code+" , "+Std.ftoa(scrollX,10)+" , "+Std.ftoa(speed,10)+" )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateScrollY(ShaderType src){
        ShaderType toReturn;
        "modulateScrollY ( tx , "+src.code+" , 0.5 , 1.0 )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateScrollY(ShaderType src, float scrollY){
        ShaderType toReturn;
        "modulateScrollY ( tx , "+src.code+" , "+Std.ftoa(scrollY,10)+" , 1.0 )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateScrollY(ShaderType src, float scrollY, float speed){
        ShaderType toReturn;
        "modulateScrollY ( tx , "+src.code+" , "+Std.ftoa(scrollY,10)+" , "+Std.ftoa(speed,10)+" )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType posterize(){
        ShaderType toReturn;
        "posterize ( tx , 3. , 0.6 )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType posterize(float bins){
        ShaderType toReturn;
        "posterize ( tx , "+Std.ftoa(bins,10)+" , 0.6 )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType posterize(float bins, float gamma){
        ShaderType toReturn;
        "posterize ( tx , "+Std.ftoa(bins,10)+" , "+Std.ftoa(gamma,10)+" )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType shift(){
        ShaderType toReturn;
        "shift ( tx , 0.1 , 0.9 , 0.3 , 1.0)" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType shift(float r){
        ShaderType toReturn;
        "shift ( tx , "+Std.ftoa(r,10)+" , 0.9 , 0.3 , 1.0 )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType shift(float r, float g){
        ShaderType toReturn;
        "shift ( tx , "+Std.ftoa(r,10)+" , "+Std.ftoa(g,10)+" , 0.3 , 1.0 )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType shift(float r, float g, float b){
        ShaderType toReturn;
        "shift ( tx , "+Std.ftoa(r,10)+" , "+Std.ftoa(g,10)+" , "+Std.ftoa(b,10)+" , 1.0 )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType shift(float r, float g, float b, float a){
        ShaderType toReturn;
        "shift ( tx , "+Std.ftoa(r,10)+" , "+Std.ftoa(g,10)+" , "+Std.ftoa(b,10)+" , "+Std.ftoa(a,10)+" )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType add(ShaderType f1){
        ShaderType toReturn;
        "add ( tx , "+f1.code+" , 0.5 )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType add(ShaderType f1, float amount){
        ShaderType toReturn;
        "add ( tx , "+f1.code+" , "+Std.ftoa(amount,10)+" )" => toReturn.code;
        "src" => toReturn.type;
        return toReturn;
    }

    function ShaderType sub(ShaderType f1){
        ShaderType toReturn;
        "sub ( tx , "+f1.code+" , 0.5 )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType sub(ShaderType f1, float amount){
        ShaderType toReturn;
        "sub ( tx , "+f1.code+" , "+Std.ftoa(amount,10)+" )" => toReturn.code;
        "src" => toReturn.type;
        return toReturn;
    }

    function ShaderType layer(ShaderType f1){
        ShaderType toReturn;
        "layer ( tx , "+f1.code+" )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType blend(ShaderType f1){
        ShaderType toReturn;
        "blend ( tx , "+f1.code+" , 0.5 )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType blend(ShaderType f1, float amount){
        ShaderType toReturn;
        "blend ( tx , "+f1.code+" , "+Std.ftoa(amount,10)+" )" => toReturn.code;
        "src" => toReturn.type;
        return toReturn;
    }

    function ShaderType mult(ShaderType f1){
        ShaderType toReturn;
        "mult ( tx , "+f1.code+" , 0.5 )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType mult(ShaderType f1, float amount){
        ShaderType toReturn;
        "mult ( tx , "+f1.code+" , "+Std.ftoa(amount,10)+" )" => toReturn.code;
        "src" => toReturn.type;
        return toReturn;
    }

    function ShaderType diff(ShaderType f1){
        ShaderType toReturn;
        "diff ( tx , "+f1.code+" , 0.5 )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType diff(ShaderType f1, float amount){
        ShaderType toReturn;
        "diff ( tx , "+f1.code+" , "+Std.ftoa(amount,10)+" )" => toReturn.code;
        "src" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulate(ShaderType src){
        ShaderType toReturn;
        "modulate ( uv , "+src.code+" , 0.5 )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateScale(ShaderType src){
        ShaderType toReturn;
        "modulateScale ( uv , "+src.code+" , 1. , 1. )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateScale(ShaderType src, float offset){
        ShaderType toReturn;
        "modulateScale ( uv , "+src.code+" , "+Std.ftoa(offset,10)+ ", 1. )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

   function ShaderType modulateScale(ShaderType src, float offset, float multiple){
        ShaderType toReturn;
        "modulateScale ( uv , "+src.code+" , "+Std.ftoa(offset,10)+ ", "+Std.ftoa(multiple,10)+" )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulatePixelate(ShaderType src){
        ShaderType toReturn;
        "modulatePixelate ( uv , "+src.code+" , 1. , 1. )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulatePixelate(ShaderType src, float offset){
        ShaderType toReturn;
        "modulatePixelate ( uv , "+src.code+" , "+Std.ftoa(offset,10)+ ", 1. )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

   function ShaderType modulatePixelate(ShaderType src, float offset, float multiple){
        ShaderType toReturn;
        "modulatePixelate ( uv , "+src.code+" , "+Std.ftoa(offset,10)+ ", "+Std.ftoa(multiple,10)+" )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateRotate(ShaderType src){
        ShaderType toReturn;
        "modulateRotate ( uv , "+src.code+" , 1. , 1. )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateRotate(ShaderType src, float offset){
        ShaderType toReturn;
        "modulateRotate ( uv , "+src.code+" , "+Std.ftoa(offset,10)+ ", 1. )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

   function ShaderType modulateRotate(ShaderType src, float offset, float multiple){
        ShaderType toReturn;
        "modulateRotate ( uv , "+src.code+" , "+Std.ftoa(offset,10)+ ", "+Std.ftoa(multiple,10)+" )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateHue(ShaderType src){
        ShaderType toReturn;
        "modulateHue ( uv , "+src.code+" , 1. , 1. )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateHue(ShaderType src, float ammount){
        ShaderType toReturn;
        "modulateHue ( uv , "+src.code+" , "+Std.ftoa(ammount,10)+ ", 1. )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType invert(){
        ShaderType toReturn;
        "invert ( tx , 1. )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType invert(float amount){
        ShaderType toReturn;
        "invert ( tx , "+Std.ftoa(amount,10)+" )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType contrast(){
        ShaderType toReturn;
        "contrast ( tx , 1.6 )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType contrast(float amount){
        ShaderType toReturn;
        "contrast ( tx , "+Std.ftoa(amount,10)+" )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType brightness(){
        ShaderType toReturn;
        "brightness ( tx , 0.4 )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType brightness(float amount){
        ShaderType toReturn;
        "brightness ( tx , "+Std.ftoa(amount,10)+" )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType mask(ShaderType f1){
        ShaderType toReturn;
        "mask ( tx , "+f1.code+" )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType luma(){
        ShaderType toReturn;
        "luma ( tx , 0.5 , 0.1 )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType luma(float threshold){
        ShaderType toReturn;
        "luma ( tx , "+Std.ftoa(threshold,10)+" , 0.1 )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType luma(float threshold, float tolerance){
        ShaderType toReturn;
        "luma ( tx , "+Std.ftoa(threshold,10)+" , "+Std.ftoa(tolerance,10)+" )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType thresh(){
        ShaderType toReturn;
        "thresh ( tx , 0.5 , 0.04 )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType thresh(float threshold){
        ShaderType toReturn;
        "thresh ( tx , "+Std.ftoa(threshold,10)+" , 0.1 )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType thresh(float threshold, float tolerance){
        ShaderType toReturn;
        "thresh ( tx , "+Std.ftoa(threshold,10)+" , "+Std.ftoa(tolerance,10)+" )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType color(){
        ShaderType toReturn;
        "color ( tx , 1. , 1. , 1. , 1. )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType color(float r){
        ShaderType toReturn;
        "color ( tx , "+Std.ftoa(r,10)+" , 1. , 1. , 1. )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType color(float r, float g){
        ShaderType toReturn;
        "color ( tx , "+Std.ftoa(r,10)+" , "+Std.ftoa(g,10)+" , 1. , 1. )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType color(float r, float g, float b){
        ShaderType toReturn;
        "color ( tx , "+Std.ftoa(r,10)+" , "+Std.ftoa(g,10)+" , "+Std.ftoa(b,10)+" , 1. )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType color(float r, float g, float b, float a){
        ShaderType toReturn;
        "color ( tx , "+Std.ftoa(r,10)+" , "+Std.ftoa(g,10)+" , "+Std.ftoa(b,10)+" , "+Std.ftoa(a,10)+" )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType r(){
        ShaderType toReturn;
        "r ( tx , 1. , 1.  )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType r(float scale){
        ShaderType toReturn;
        "r ( tx , "+Std.ftoa(scale,10)+" , 1.  )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType r(float scale, float offset){
        ShaderType toReturn;
        "r ( tx , "+Std.ftoa(scale,10)+" , "+Std.ftoa(offset,10)+"  )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType g(){
        ShaderType toReturn;
        "g ( tx , 1. , 1.  )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType g(float scale){
        ShaderType toReturn;
        "g ( tx , "+Std.ftoa(scale,10)+" , 1.  )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType g(float scale, float offset){
        ShaderType toReturn;
        "g ( tx , "+Std.ftoa(scale,10)+" , "+Std.ftoa(offset,10)+"  )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType b(){
        ShaderType toReturn;
        "b ( tx , 1. , 1.  )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType b(float scale){
        ShaderType toReturn;
        "b ( tx , "+Std.ftoa(scale,10)+" , 1.  )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType b(float scale, float offset){
        ShaderType toReturn;
        "b ( tx , "+Std.ftoa(scale,10)+" , "+Std.ftoa(offset,10)+"  )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType a(){
        ShaderType toReturn;
        "a ( tx , 1. , 1.  )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType a(float scale){
        ShaderType toReturn;
        "a ( tx , "+Std.ftoa(scale,10)+" , 1.  )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType a(float scale, float offset){
        ShaderType toReturn;
        "a ( tx , "+Std.ftoa(scale,10)+" , "+Std.ftoa(offset,10)+"  )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType saturate(){
        ShaderType toReturn;
        "saturate ( tx , 2.  )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType saturate(float amount){
        ShaderType toReturn;
        "saturate ( tx , "+Std.ftoa(amount,10)+" )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType hue(){
        ShaderType toReturn;
        "hue ( tx , 2.  )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType hue(float amount){
        ShaderType toReturn;
        "hue ( tx , "+Std.ftoa(amount,10)+" )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType colorama(){
        ShaderType toReturn;
        "colorama ( tx , 2.  )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType colorama(float amount){
        ShaderType toReturn;
        "colorama ( tx , "+Std.ftoa(amount,10)+" )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateSR(ShaderType src){
        ShaderType toReturn;
        "modulateSR ( uv , "+src.code+" , 4. , 0.5 , 2.0 , 0.5 )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateSR(ShaderType src, float multiple){
        ShaderType toReturn;
        "modulateSR ( uv , "+src.code+" , "+Std.ftoa(multiple,10)+" , 0.5 , 2.0 , 0.5 )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateSR(ShaderType src, float multiple, float offset){
        ShaderType toReturn;
        "modulateSR ( uv , "+src.code+" , "+Std.ftoa(multiple,10)+" , "+Std.ftoa(offset,10)+" , 2.0 , 0.5 )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateSR(ShaderType src, float multiple, float offset, float rotateMultiple){
        ShaderType toReturn;
        "modulateSR ( uv , "+src.code+" , "+Std.ftoa(multiple,10)+" , "+Std.ftoa(offset,10)+" , "+Std.ftoa(rotateMultiple,10)+" , 0.5 )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateSR(ShaderType src, float multiple, float offset, float rotateMultiple, float rotateOffset){
        ShaderType toReturn;
        "modulateSR ( uv , "+src.code+" , "+Std.ftoa(multiple,10)+" , "+Std.ftoa(offset,10)+" , "+Std.ftoa(rotateMultiple,10)+" , "+Std.ftoa(rotateOffset,10)+" )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType chroma(){
        ShaderType toReturn;
        "chroma ( tx )" => toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType sphere(){
        ShaderType toReturn;
        "sphere ( uv , 2. , 2. )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType sphere(float radius){
        ShaderType toReturn;
        "sphere ( uv , "+Std.ftoa(radius,10)+" , 2. )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType sphere(float radius, float rot){
        ShaderType toReturn;
        "sphere ( uv , "+Std.ftoa(radius,10)+" , "+Std.ftoa(rot,10)+" )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType sphereDisplacement(ShaderType src){
        ShaderType toReturn;
        "sphereDisplacement ( uv , "+src.code+" , 2. , 2. )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType sphereDisplacement(ShaderType src, float radius){
        ShaderType toReturn;
        "sphereDisplacement ( uv , "+src.code+" , "+Std.ftoa(radius,10)+" , 2. )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType sphereDisplacement(ShaderType src, float radius, float rot){
        ShaderType toReturn;
        "sphereDisplacement ( uv , "+src.code+" , "+Std.ftoa(radius,10)+" , "+Std.ftoa(rot,10)+" )" => toReturn.code;
        "trns" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateRays(){
        ShaderType toReturn;
        "modulateRays ( uv , tx , 1.0 , 100. )"=> toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateRays(float scale){
        ShaderType toReturn;
        "modulateRays ( uv , tx , "+Std.ftoa(scale,10)+" , 100. )"=> toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    function ShaderType modulateRays(float scale, float samples){
        ShaderType toReturn;
        "modulateRays ( uv , tx , "+Std.ftoa(scale,10)+" , "+Std.ftoa(samples,10)+" )"=> toReturn.code;
        "trnsC" => toReturn.type;
        return toReturn;
    }

    //--------------------------------------
    function string shader(string code){
        return header+code+close;
    }

    function string shader(ShaderType singleObject){
        // <<< "code:",singleObject.code >>>;
        // shaderMat.fragShader( header+singleObject.code+close );
        return header+singleObject.code+close;
    }
}

//----------------------------------------------------------------

function ShaderType CodeMaker(ShaderType one, ShaderType two, string type){

    StringTokenizer tokenizer;

    tokenizer.set( one.code );

    ShaderType toReturn;
    type => toReturn.type;

    while( tokenizer.more() )
    {
        tokenizer.next() => string temp;

        if(type == "src"){
            if(temp == "uv"){
                two.code +=> toReturn.code;
            } else {
                temp +=> toReturn.code; 
            }
        }
        if(type == "srcC"){
            if(temp == "tx"){
                two.code +=> toReturn.code;
            } else {
                temp +=> toReturn.code; 
            }
        }
    }

    "src" => toReturn.type;
    // <<< toReturn.code >>>;
    return toReturn;
}

public ShaderType @operator ->(ShaderType one ,ShaderType two){

    if(one.type == "src" && two.type == "trns"){
        return CodeMaker(one, two, "src");
    }
    if(one.type == "src" && two.type == "trnsC"){
        return CodeMaker(two, one, "srcC");
    }

    return ShaderType n;
}

GG.windowTitle( "Code Punk Alchemy" );

eon => now;
 // This code is pretty much a collection of GLSL helper functions from multiple sources
 // plus the hydra by Olivia Jack and some other spices added by Celeste BEtancur Gutierrez
 
 "
 #version 330 core
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
        return fract(_st);
    }

    vec2 scrollX (vec2 _st, float scrollX, float speed){
        _st.x += scrollX + u_Time*speed;
        return fract(_st);
    }

    vec2 modulateScrollX (vec2 _st, vec4 _c0, float scrollX, float speed){
        _st.x += _c0.r*scrollX + u_Time*speed;
        return fract(_st);
    }

    vec2 scrollY (vec2 _st, float scrollY, float speed){
        _st.y += scrollY + u_Time*speed;
        return fract(_st);
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
        c2.r += fract(r);
        c2.g += fract(g);
        c2.b += fract(b);
        c2.a += fract(a);
        return c2.rgba;
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
        FragColor = " => global string hydra;


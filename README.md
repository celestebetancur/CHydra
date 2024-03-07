# CHydra LiveCodingVisuals inside ChucK
Code library and examples 


##1. Install chuck in your favorite flavor
#
https://chuck.stanford.edu/release/

##2. Download and install ChuGL
#
https://chuck.stanford.edu/chugl/

##3. Download the contents of this repo and execute the main game loop:
#
chuck --loop 1.ck

##4. Make changes to playground using miniAudicle (or any code editor) and do:
#
chuck + playground.ck

If you are feeling spicy and adventurous:

chuck --loop 2.ck
chuck + playground2.ck

//This command will give you the IDs of the current shreds:
chuck ^ 

//Make changes in playground2.ck and then do
chuck = ID playground2.ck
// where ID is the ID number you find with status 
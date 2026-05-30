from pymol import cmd,stored

set depth_cue, 1
set fog_start, 0.4

set_color b_col, [36,36,85]
set_color t_col, [10,10,10]
set bg_rgb_bottom, b_col
set bg_rgb_top, t_col      
set bg_gradient

set  spec_power  =  200
set  spec_refl   =  0

load "data/NAMPT.pdb", protein
create ligands, protein and organic
select xlig, protein and organic
delete xlig

hide everything, all

color white, elem c
color bluewhite, protein
#show_as cartoon, protein
show surface, protein
#set transparency, 0.15

show sticks, ligands
set stick_color, magenta




# SAS points

load "data/NAMPT.pdb_points.pdb.gz", points
hide nonbonded, points
show nb_spheres, points
set sphere_scale, 0.2, points
cmd.spectrum("b", "green_red", selection="points", minimum=0, maximum=0.7)


stored.list=[]
cmd.iterate("(resn STP)","stored.list.append(resi)")    # read info about residues STP
lastSTP=stored.list[-1] # get the index of the last residue
hide lines, resn STP

cmd.select("rest", "resn STP and resi 0")

for my_index in range(1,int(lastSTP)+1): cmd.select("pocket"+str(my_index), "resn STP and resi "+str(my_index))
for my_index in range(1,int(lastSTP)+1): cmd.show("spheres","pocket"+str(my_index))
for my_index in range(1,int(lastSTP)+1): cmd.set("sphere_scale","0.4","pocket"+str(my_index))
for my_index in range(1,int(lastSTP)+1): cmd.set("sphere_transparency","0.1","pocket"+str(my_index))



set_color pcol1 = [0.361,0.576,0.902]
select surf_pocket1, protein and id [1508,1510,2740,2743,2993,2994,2997,2998,2989,2990,2778,2789,3032,3033,3034,2771,2772,2773,2774,2775,2395,2967,2722,2723,2724,2957,2963,2941,2694,2725,2726,6828,2416,2418,1902,2762,2764,2437,2438,6838,6839,6847,6848,6849,2769,2770,7052,7075,7076,6834,6835,6842,6843,6867,6877,6872,6846,6873,1695,1487,1488,1489,1490,1491,1492,1509,1694,1511,1846,1856,1860,2122,2123,1866,1867,2396,1864,4414,2108,2109,2363,2379,2380,2110,2356,2359,1504,3864,1535,3863,1502,1505,1506,1507,1889,3881,3882,1879,3879,1536,4920,1891,1892,3880,1533,1534,4921,1421,1472,1473,1474,1413,2942,1448,1449,1678,1419,1450,1452,1454,1456,1458,1460,1462,1847,4437,1848,1849,1850,1851,1852,1853,1900,6902,6903,6904,6905,7017,7071,7073,4072,4074,6885,4890,1893,1894,1904,1908,1898,6886,6889,6898,6906] 
set surface_color,  pcol1, surf_pocket1 
set_color pcol2 = [0.278,0.278,0.702]
select surf_pocket2, protein and id [5286,5289,5290,5292,5271,5273,5274,5275,5276,5291,5293,5295,5294,6524,5241,5651,6185,5897,5898,6152,6169,6145,6168,6770,6771,6774,6775,6776,5324,5288,6521,6523,6184,6748,6513,6736,6738,5258,6730,6731,6503,6504,6505,6506,6507,6722,6148,6475,6473,6476,6558,6559,6552,6553,6554,6543,6556,6778,6567,6779,6813,6815,6243,6244,3054,3057,5317,3058,3096,5318,5319,3065,5320,6217,6218,6219,79,6201,6202,5663,5911,5912,3236,5684,5686,5688,5682,5673,5670,5678,3100,3104,3124,275,273,3108,3121,3123,3066,3067,3068,6550,3294,3295,6229,6545,6551,277,3292,3288,3290,3291,3271,3061,300,3048,3062,299,3092,3047,3049,5478,5630,5476,5644,5462,5639,5640,5631,5641,5642,5650,5899,5633,637,638,651,5323,5479,5648,74,77,62,78,80,1131,1130,5676,5242,5243,5244,5245,5246,5248,5220,5247,5218,5256,5257,5212,6723,6724,6727,5632,5635,5636,5637] 
set surface_color,  pcol2, surf_pocket2 
set_color pcol3 = [0.576,0.361,0.902]
select surf_pocket3, protein and id [5350,5353,1,2,5,7,12,5497,15,38,39,40,5515,16,42,5348,5349,5355,5356,5498,5495,48,45,5547,5514,5559,5377,4203,5520,5522,5516,5517,5360] 
set surface_color,  pcol3, surf_pocket3 
set_color pcol4 = [0.616,0.278,0.702]
select surf_pocket4, protein and id [1572,1576,448,449,1738,413,1775,1764,1763,1713,1729,1730,1731,1732,1733,1736,1737,1593,1566,1569,3809,3811,1714,3817,3803,3804,3807,3814,1564,1565,3841,3842,3844,4351,1706,1711] 
set surface_color,  pcol4, surf_pocket4 
set_color pcol5 = [0.902,0.361,0.792]
select surf_pocket5, protein and id [3358,3359,3376,3485,3349,869,3551,3554,3486,3487,3488,3494,867,3339,3343,3342,218,989,1006,217,1005] 
set surface_color,  pcol5, surf_pocket5 
set_color pcol6 = [0.702,0.278,0.447]
select surf_pocket6, protein and id [7130,7139,7140,7120,4019,4779,7123,4796,4653,4654,4655,7335,7332,7124,7275,7267,7268,7269,7270,7325,7327,7326,7266,7157] 
set surface_color,  pcol6, surf_pocket6 
set_color pcol7 = [0.902,0.361,0.361]
select surf_pocket7, protein and id [3924,3926,4481,4482,4830,4850,3920,4852,4834,3962,3967,3973] 
set surface_color,  pcol7, surf_pocket7 
set_color pcol8 = [0.702,0.447,0.278]
select surf_pocket8, protein and id [160,165,1062,182,1060,1058,148,118,128,124,122,696,1040,710,711,712,716,1044,695] 
set surface_color,  pcol8, surf_pocket8 
set_color pcol9 = [0.902,0.792,0.361]
select surf_pocket9, protein and id [6571,6577,5054,6585,6781,6823,5055,6810,4116,4117,6857,6856,6567,6814,6570] 
set surface_color,  pcol9, surf_pocket9 
   

deselect

orient

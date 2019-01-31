'''
ECE276A WI19 HW1
Blue Barrel Detector
'''

import os
import cv2
from skimage.measure import label, regionprops
from skimage import data, util, img_as_ubyte
from skimage.morphology import erosion, dilation, opening, closing, disk
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math
import train

class BarrelDetector(object):
	def __init__(self):
		'''
			Initialize your blue barrel detector with the attributes you need
			eg. parameters of your classifier
		'''
		
	def segment_image(self, img):
		'''
			Calculate the segmented image using a classifier
			eg. Single Gaussian, Gaussian Mixture, or Logistic Regression
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				mask_img - a binary image with 1 if the pixel in the original image is blue and 0 otherwise
		'''
		#use model to find blue parts of image
		model = train.LR_Model()
		#model.load('weights.pickle')
		model.weights = np.array([[0.004316421032463076, -0.0043164210324630765], [0.0035589924053128887, -0.003558992405312892], [0.002931370840072359, -0.00293137084007236], [0.0040341439089877865, -0.004034143908987793], [0.002772463008034349, -0.0027724630080343433], [0.004780432801345808, -0.004780432801345803], [0.0040784197213062125, -0.004078419721306218], [0.003687565396011121, -0.003687565396011125], [0.003427568403489808, -0.00342756840348981], [0.002342825739991563, -0.0023428257399915648], [0.0012550121123091425, -0.0012550121123091425], [0.0021010545239288345, -0.0021010545239288415], [0.0031727364640076875, -0.003172736464007689], [0.0037429228004955007, -0.003742922800495499], [0.005061349196098476, -0.005061349196098482], [0.0031988692619133157, -0.0031988692619133174], [0.000746732658793382, -0.0007467326587933836], [0.0022041336640427575, -0.0022041336640427588], [0.001153674896242036, -0.0011536748962420373], [0.0007865348845042641, -0.0007865348845042615], [0.0009281625599946279, -0.0009281625599946269], [0.0011311843154352362, -0.0011311843154352364], [0.002490683155119205, -0.002490683155119204], [0.0008104393810992527, -0.0008104393810992546], [0.00011864198567846938, -0.00011864198567846833], [-0.0012164379644071938, 0.0012164379644071927], [9.258448495652363e-06, -9.258448495649588e-06], [0.0010508670276452123, -0.0010508670276452125], [0.0006917727519999498, -0.0006917727519999512], [0.003935353638465439, -0.003935353638465445],
		[0.002553854530662062, -0.002553854530662066], [0.0011435447568380168, -0.0011435447568380196], [-5.129131796111648e-05, 5.1291317961117444e-05], [0.0005209460326399052, -0.0005209460326399051], [0.0007080900373888588, -0.0007080900373888583], [0.002442837749962989, -0.002442837749962992], [0.001664485134107005, -0.001664485134107007], [0.001979008656312139, -0.0019790086563121405], [0.0010735490123640345, -0.001073549012364036], [-0.0008276053994165574, 0.0008276053994165564], [-0.0009289869130593473, 0.000928986913059349], [0.0003394817478758468, -0.0003394817478758506], [0.0023138970903531218, -0.0023138970903531287], [0.0015679485710166996, -0.0015679485710167012], [0.0033393625898816086, -0.0033393625898816095], [0.0016238126790783616, -0.001623812679078368], [-0.0006920902155568086, 0.0006920902155568096], [-0.001690058747899121, 0.0016900587478991223], [0.001512109510592646, -0.0015121095105926439], [0.0008338242993641926, -0.0008338242993641945], [-0.0001658508326568584, 0.00016585083265685657], [0.0025283099184524763, -0.002528309918452474], [0.00034225464215777407, -0.0003422546421577741], [-0.0004274558543697158, 0.00042745585436971764], [-0.001996254382668634, 0.0019962543826686353], [0.0005060825827567668, -0.0005060825827567658], [0.001282023708367828, -0.0012820237083678275], [0.00020511826626840913, -0.00020511826626840965], [0.0018328620815074592, -0.00183286208150746], [0.0019444794204826837, -0.0019444794204826834],
		[0.00126985177908044, -0.0012698517790804386], [0.0009008429436937324, -0.0009008429436937323], [0.00032865386709656457, -0.0003286538670965631], [0.00268549834452226, -0.0026854983445222555], [0.0017220740764502611, -0.0017220740764502605], [0.0013537456762531274, -0.0013537456762531303], [0.0034107467018266707, -0.0034107467018266746], [0.0009804447268576555, -0.0009804447268576557], [0.001313615723241944, -0.0013136157232419446], [-0.0002448933097931869, 0.00024489330979318865], [0.001718672899147853, -0.0017186728991478504], [0.0027995995254065363, -0.002799599525406534], [0.00221246290332327, -0.0022124629033232713], [0.0049368764604978245, -0.004936876460497823], [0.002874481860880809, -0.0028744818608808023], [-0.0008403347248343801, 0.0008403347248343805], [-0.000798357752443378, 0.0007983577524433775], [-0.0003853597009095205, 0.00038535970090952003], [-1.0389152556355464e-05, 1.0389152556356968e-05], [-0.002252300403156746, 0.0022523004031567468], [-0.0009481242451194199, 0.000948124245119421], [0.001589597964995379, -0.0015895979649953797], [0.00011687844847457245, -0.00011687844847457201], [-0.0009659091529536362, 0.0009659091529536376], [-0.0021142853290553104, 0.0021142853290553112], [-0.002101819862871586, 0.0021018198628715862], [-0.0020801561595383184, 0.0020801561595383184], [-0.0008088012085278554, 0.0008088012085278544], [0.0007776329287275606, -0.0007776329287275601], [0.00037724527860460825, -0.00037724527860460717],
		[0.00016950286885798882, -0.0001695028688579881], [-8.325412741682728e-05, 8.325412741682617e-05], [-0.00012111323961268424, 0.00012111323961268302], [-0.0006742893969577895, 0.0006742893969577903], [-0.001516345954364083, 0.0015163459543640792], [0.001123467011876454, -0.0011234670118764531], [0.0019822811229337324, -0.0019822811229337346], [0.0008671757934060091, -0.000867175793406011], [0.0010232558797892578, -0.0010232558797892588], [-0.00036119056748680527, 0.00036119056748680375], [-0.00140897427211293, 0.001408974272112928], [-0.0007725316552548218, 0.0007725316552548224], [0.0003970291018768635, -0.00039702910187686336], [0.0011250095810900135, -0.0011250095810900163], [0.0007925212861618313, -0.0007925212861618323], [4.124858161783873e-05, -4.124858161784021e-05], [0.00031802527039937237, -0.00031802527039937427], [0.0010747368312981447, -0.001074736831298143], [0.0019360099274892044, -0.0019360099274892055], [0.0004972780785701766, -0.0004972780785701749], [0.001732929113707168, -0.0017329291137071702], [0.0029266717899111656, -0.0029266717899111643], [0.0021357056882849636, -0.0021357056882849675], [0.0020032685915365722, -0.0020032685915365753], [0.0007877884992028339, -0.0007877884992028327], [2.8760363925020713e-05, -2.876036392502061e-05], [0.0003551532111826714, -0.00035515321118267213], [8.841904441753884e-05, -8.841904441753921e-05], [0.0016521365349865037, -0.0016521365349865035], [0.0015150423118156103, -0.00151504231181561],
		[0.0016932310154686744, -0.0016932310154686718], [0.00026624831847096647, -0.00026624831847096516], [0.0022766787795216, -0.0022766787795216003], [0.0019641158388740815, -0.0019641158388740824], [0.0014974362285120513, -0.001497436228512052], [0.0021884256287563355, -0.002188425628756339], [0.00373342247200398, -0.003733422472003978], [0.0042574838978289755, -0.004257483897828978], [0.0035215082339422814, -0.0035215082339422766], [0.0027206583669240262, -0.0027206583669240284], [0.0007541621642954599, -0.0007541621642954591], [0.0015195162011255082, -0.0015195162011255078], [0.0012775379023530324, -0.0012775379023530335], [0.001157171491205955, -0.0011571714912059564], [0.002637173017467132, -0.0026371730174671337], [0.00014508350431644636, -0.0001450835043164464], [0.0001289656955588115, -0.00012896569555881314], [5.610514570887241e-05, -5.6105145708874185e-05], [0.00036524158459890936, -0.00036524158459891017], [-0.00028817490931570626, 0.00028817490931570507], [0.002101326245535739, -0.0021013262455357386], [0.0028253520380782556, -0.0028253520380782577], [0.00262859227221249, -0.0026285922722124855], [0.0028928553506071793, -0.0028928553506071797], [0.0008884532920318658, -0.0008884532920318686], [0.0010358201465363753, -0.0010358201465363772], [0.0015270334211650027, -0.0015270334211650064], [0.0022763998645762337, -0.002276399864576235], [0.001898206803609691, -0.0018982068036096909], [0.0017559236918459147, -0.0017559236918459175],
		[-0.001380604046040307, 0.0013806040460403086], [-0.002658912973597273, 0.0026589129735972725], [-0.001942866190101817, 0.0019428661901018177], [0.0015377531530183695, -0.001537753153018368], [-2.5037933720592975e-05, 2.50379337205934e-05], [-3.5377383553999105e-05, 3.537738355399774e-05], [0.002710113499143372, -0.0027101134991433704], [0.0007142490133794292, -0.0007142490133794288], [-0.00025225436755847377, 0.00025225436755847285], [-0.0019137578224879429, 0.001913757822487941], [0.001091598923879684, -0.001091598923879679], [0.0004586074815330626, -0.0004586074815330615], [0.0003687812628421311, -0.00036878126284213103], [0.0018265584531467487, -0.0018265584531467498], [0.00032268488709108576, -0.00032268488709108527], [-0.0022481269371121475, 0.002248126937112147], [-0.001979351151055973, 0.0019793511510559736], [-0.0015022233277865334, 0.0015022233277865314], [0.0008763114918310017, -0.0008763114918310007], [-0.00014781163394346917, 0.00014781163394346865], [-7.073442761321623e-05, 7.07344276132129e-05], [0.002489503114656719, -0.0024895031146567204], [6.957709013604082e-05, -6.95770901360387e-05], [-0.0004632510823853531, 0.00046325108238535307], [-0.0021185319193481105, 0.002118531919348106], [0.000382423363137842, -0.0003824233631378412], [0.0011459014292706396, -0.0011459014292706448], [0.0003360217168951494, -0.00033602171689515027], [0.0028768929685415523, -0.0028768929685415497], [0.0011192490852655087, -0.0011192490852655083],
		[-0.00279636200246532, 0.0027963620024653193], [-0.0014854221344030502, 0.0014854221344030522], [-2.41164937375421e-05, 2.4116493737541006e-05], [0.0005254951516955405, -0.0005254951516955426], [-0.0007560904838788833, 0.0007560904838788835], [4.4078459805107794e-05, -4.407845980510757e-05], [0.001991924954500583, -0.001991924954500582], [0.0009647041464818782, -0.0009647041464818758], [0.001011558084669199, -0.001011558084669201], [-0.0001122769312683202, 0.00011227693126832051], [-0.00039644459067806725, 0.00039644459067806584], [0.00016751012379754245, -0.0001675101237975448], [0.0008220984173319515, -0.0008220984173319528], [0.0014850501090890149, -0.0014850501090890149], [-0.0005056613942144571, 0.000505661394214452], [-0.0031448708683296913, 0.0031448708683296965], [-0.001010382471524729, 0.0010103824715247268], [-0.00029810838783644736, 0.0002981083878364431], [-0.00044973650050108454, 0.0004497365005010819], [-0.0015237315466592558, 0.0015237315466592562], [0.0008774227498796491, -0.000877422749879649], [0.0021863584409223488, -0.002186358440922349], [0.000929192075128913, -0.0009291920751289119], [0.0013719100973830474, -0.0013719100973830507], [0.000697764517783659, -0.0006977645177836585], [-0.0004521589475277369, 0.00045215894752773626], [-0.0006285463361769695, 0.0006285463361769701], [0.0013669496023991045, -0.0013669496023991055], [0.0011095647021858322, -0.0011095647021858277], [-0.0002306248561129193, 0.00023062485611292116],
		[-0.006025242584744146, 0.006025242584744147], [-0.0033410942994239754, 0.0033410942994239767], [-0.0027911295081924563, 0.0027911295081924537], [-0.0015293099272279031, 0.0015293099272279001], [-0.0027740820784631403, 0.002774082078463141], [-0.0010102408663281968, 0.00101024086632819], [1.1700152617405435e-05, -1.1700152617406432e-05], [-4.8579318226790775e-06, 4.857931822678677e-06], [0.0004020680943658277, -0.00040206809436582575], [-0.0003029802128396837, 0.0003029802128396821], [-0.0006035129309398291, 0.0006035129309398268], [-0.0017104845370203808, 0.00171048453702038], [-0.0020350472885891017, 0.0020350472885890974], [-0.0023776737855715703, 0.002377673785571566], [-0.0025759836012130667, 0.0025759836012130676], [0.0028868933361867136, -0.0028868933361867197], [0.0017455568924440602, -0.0017455568924440604], [-0.0007123426886140521, 0.0007123426886140513], [-0.002882344343144866, 0.002882344343144864], [-0.005736836920403487, 0.005736836920403493], [-0.008964542550195076, 0.00896454255019508], [-0.011747049031877649, 0.01174704903187765], [-0.011894874748206873, 0.011894874748206883], [-0.00931138361625289, 0.009311383616252871], [-0.005680688871854681, 0.005680688871854684], [-0.0007981205512213806, 0.0007981205512213778], [0.00037075275233536725, -0.00037075275233536275], [0.0004225996759975644, -0.00042259967599756653], [0.0017350773863173978, -0.0017350773863173902], [0.005556069219359515, -0.005556069219359518],
		[0.0025925963273095575, -0.0025925963273095588], [0.0008089147899920681, -0.0008089147899920701], [0.00024156509765851234, -0.00024156509765851515], [-0.0012780855597691503, 0.001278085559769147], [-0.00411731483115379, 0.004117314831153793], [-0.007260897276663722, 0.007260897276663727], [-0.009502776138524623, 0.009502776138524623], [-0.010011296178669082, 0.010011296178669086], [-0.008188189493626593, 0.008188189493626591], [-0.0034905985149397325, 0.0034905985149397342], [0.0005273236918300401, -0.0005273236918300406], [0.000744933441966532, -0.0007449334419665332], [0.00033185027321051114, -0.00033185027321051174], [0.00122915664643129, -0.0012291566464312817], [0.005032088252468652, -0.005032088252468644], [0.0024507677562162562, -0.0024507677562162597], [0.0013514051818976824, -0.001351405181897686], [0.0005628806859140259, -0.0005628806859140268], [-0.0009045571304440121, 0.000904557130444014], [-0.0030732443630994604, 0.003073244363099461], [-0.0061370123179304795, 0.006137012317930479], [-0.0077340611829945435, 0.007734061182994554], [-0.00780285466948809, 0.007802854669488098], [-0.006156670536234153, 0.006156670536234148], [-0.0025508920819973186, 0.0025508920819973134], [0.0006850049357398741, -0.0006850049357398789], [0.0011883953164306925, -0.0011883953164306923], [0.00038241578234172405, -0.00038241578234172286], [0.0019572936581411715, -0.0019572936581411784], [0.005427162407981359, -0.0054271624079813615],
		[0.001870579500646504, -0.0018705795006465082], [0.002199426274322203, -0.002199426274322208], [0.0010941019268475825, -0.001094101926847581], [-0.00044836311577464596, 0.00044836311577464526], [-0.0029770414213130484, 0.0029770414213130454], [-0.006042531681221782, 0.006042531681221785], [-0.007268631333961397, 0.007268631333961408], [-0.00717941079951721, 0.007179410799517218], [-0.004387333096116459, 0.004387333096116457], [-0.001965284706990803, 0.001965284706990799], [0.00043761302622120213, -0.0004376130262212029], [0.0003923525529537628, -0.00039235255295376253], [-0.0001969540804199851, 0.00019695408041998396], [0.002124403733676677, -0.0021244037336766736], [0.005340545764013346, -0.005340545764013349], [0.0008739991771691192, -0.0008739991771691191], [0.0001738345424337354, -0.00017383454243373564], [0.00014570333276266355, -0.0001457033327626668], [-0.002099329508282099, 0.0020993295082821013], [-0.004651015583939909, 0.004651015583939899], [-0.0070348595438386174, 0.00703485954383863], [-0.008287614039426166, 0.008287614039426162], [-0.008034158423273097, 0.008034158423273104], [-0.006582751487108275, 0.006582751487108271], [-0.002923871711690535, 0.0029238717116905353], [-0.0012006390466906254, 0.001200639046690631], [-0.0011910562071483159, 0.0011910562071483128], [-0.0011798838619388333, 0.0011798838619388335], [0.001059302204785112, -0.001059302204785108], [0.004311154049243198, -0.0043111540492431965],
		[0.000366142936269619, -0.0003661429362696227], [-0.000579991362324588, 0.0005799913623245862], [-0.001221173272078408, 0.0012211732720784048], [-0.0028244169420590377, 0.002824416942059036], [-0.005384280293236972, 0.005384280293236982], [-0.007644053645848208, 0.007644053645848211], [-0.00922308767689524, 0.009223087676895232], [-0.008959353109748233, 0.008959353109748224], [-0.008413498688581334, 0.008413498688581343], [-0.005208794768618742, 0.005208794768618753], [-0.001888781267285559, 0.0018887812672855576], [-0.0018900184231569737, 0.0018900184231569782], [-0.0019139798765931196, 0.0019139798765931203], [-0.0006194594524277938, 0.0006194594524277921], [0.003251465566719675, -0.0032514655667196667], [0.0002443156981039451, -0.0002443156981039448], [-0.0004347733750618378, 0.0004347733750618365], [-0.002153897121983722, 0.0021538971219837275], [-0.0038960179562025797, 0.003896017956202573], [-0.005403404787479656, 0.005403404787479654], [-0.0072829299437061875, 0.007282929943706186], [-0.009855905346017062, 0.009855905346017067], [-0.01013060643196826, 0.01013060643196826], [-0.009130937737156635, 0.009130937737156628], [-0.006511941170186002, 0.006511941170186], [-0.002815943876666424, 0.002815943876666421], [-0.0018834332991770832, 0.0018834332991770884], [-0.0017258208031492351, 0.0017258208031492334], [-0.0008494067048158508, 0.0008494067048158512], [0.0023715648723783363, -0.0023715648723783315],
		[-0.0006139014611332888, 0.0006139014611332885], [-0.000657957540376344, 0.0006579575403763445], [-0.0022768396423960787, 0.0022768396423960675], [-0.003843208325534081, 0.0038432083255340683], [-0.005583938672115962, 0.005583938672115953], [-0.007656170005389932, 0.0076561700053899345], [-0.010087973890569908, 0.01008797389056991], [-0.010935893654908388, 0.010935893654908383], [-0.009475778954971908, 0.009475778954971925], [-0.006909508010609013, 0.0069095080106090145], [-0.0031484711349547227, 0.003148471134954721], [-0.0023942524989484524, 0.0023942524989484498], [-0.0027057009794594087, 0.0027057009794594065], [-0.0017565507154043087, 0.0017565507154043074], [0.001096905843268048, -0.0010969058432680513], [-0.0009757180133471382, 0.0009757180133471364], [-0.0016872891972387087, 0.0016872891972387074], [-0.0017135460229954602, 0.0017135460229954654], [-0.002961476364772053, 0.002961476364772056], [-0.0056803348963420704, 0.005680334896342074], [-0.00822523613982737, 0.008225236139827369], [-0.010349031765277003, 0.010349031765277003], [-0.011137565816596471, 0.01113756581659649], [-0.010576472480391256, 0.01057647248039125], [-0.006641676626129715, 0.006641676626129728], [-0.0032706628821448417, 0.003270662882144845], [-0.0032461183737508984, 0.003246118373750902], [-0.003689551445993732, 0.003689551445993734], [-0.0025522436553983897, 0.00255224365539839], [0.000692235879274558, -0.0006922358792745575],
		[-0.0012043603795272519, 0.0012043603795272517], [-0.0013432345708746128, 0.0013432345708746108], [-0.0011589405101357219, 0.0011589405101357269], [-0.002687143642297942, 0.0026871436422979404], [-0.0046503642557318686, 0.004650364255731873], [-0.007472469745940205, 0.0074724697459402035], [-0.009237600497568738, 0.009237600497568749], [-0.010391010599588232, 0.010391010599588238], [-0.010114211749174646, 0.010114211749174653], [-0.006646686427220606, 0.006646686427220597], [-0.003938532441178419, 0.003938532441178411], [-0.0030495403060926754, 0.003049540306092682], [-0.0034728018663017367, 0.003472801866301738], [-0.0017716047557345667, 0.0017716047557345647], [0.0013178137401372721, -0.0013178137401372724], [-0.0008434135109264386, 0.0008434135109264393], [0.0003431983690702069, -0.00034319836907021256], [-0.00014292685066599176, 0.00014292685066599094], [-0.0013931501848919972, 0.0013931501848919914], [-0.003823895590758191, 0.003823895590758196], [-0.006742780867891009, 0.006742780867891014], [-0.007581194837828155, 0.007581194837828163], [-0.009017081746766022, 0.00901708174676602], [-0.007474034086204315, 0.007474034086204315], [-0.005187981828941876, 0.005187981828941879], [-0.0025964435440109313, 0.0025964435440109243], [-0.0020984683804579486, 0.00209846838045794], [-0.002498166640526785, 0.002498166640526789], [-0.0002371861322064288, 0.00023718613220642933], [0.0020555051775164573, -0.0020555051775164542],
		[0.00046110003917187287, -0.00046110003917187374], [0.0008188031193350027, -0.0008188031193350018], [0.0013409705301292414, -0.0013409705301292423], [-0.0007809748743552214, 0.000780974874355223], [-0.0026360321242957195, 0.0026360321242957234], [-0.004909165681810645, 0.004909165681810651], [-0.006243923810190927, 0.006243923810190918], [-0.006951369383874426, 0.006951369383874417], [-0.00646573207495022, 0.006465732074950221], [-0.003243421568921623, 0.003243421568921625], [-0.0012810040369217153, 0.0012810040369217218], [-0.0006901290134593502, 0.0006901290134593486], [-0.0008078673243413886, 0.0008078673243413878], [0.0009445885297641776, -0.0009445885297641786], [0.0032903616644112486, -0.0032903616644112525], [0.0005200075608769419, -0.0005200075608769387], [0.0007543298045412391, -0.0007543298045412382], [0.0005420881738805221, -0.000542088173880523], [-0.0002541604700738191, 0.0002541604700738191], [-0.0019304606679494212, 0.0019304606679494273], [-0.0035900556927507053, 0.0035900556927507023], [-0.005170378247778695, 0.005170378247778698], [-0.00613624487486781, 0.006136244874867806], [-0.00628540029435922, 0.0062854002943592245], [-0.0037458877974159577, 0.0037458877974159586], [-0.0007639340270634316, 0.0007639340270634333], [-0.0001941853542778023, 0.00019418535427780247], [-0.00046283215812623594, 0.0004628321581262351], [0.0007514873460917937, -0.0007514873460917954], [0.0038151139103238425, -0.003815113910323841],
		[0.00036458222533974997, -0.0003645822253397551], [0.0006609711281206341, -0.0006609711281206362], [-7.665467483211308e-05, 7.665467483211117e-05], [-0.0009616640305941932, 0.0009616640305941913], [-0.0020639563508843706, 0.0020639563508843693], [-0.003660894769532594, 0.0036608947695325987], [-0.006086335163994126, 0.00608633516399413], [-0.007081275241535524, 0.007081275241535523], [-0.00689647010656857, 0.006896470106568577], [-0.004477325438192091, 0.004477325438192096], [-0.0010347762665753504, 0.0010347762665753467], [-6.837550013805362e-05, 6.837550013805374e-05], [-0.00021125182970641455, 0.00021125182970641482], [0.0007228654384168668, -0.0007228654384168674], [0.003752027943355812, -0.0037520279433558117], [-0.0009277675614310951, 0.0009277675614310863], [-0.00034464704821501787, 0.0003446470482150172], [-0.001305275892860213, 0.001305275892860213], [-0.0020045883704747527, 0.002004588370474748], [-0.0037970315963937787, 0.0037970315963937714], [-0.0056884324406899205, 0.005688432440689925], [-0.008407521063212029, 0.008407521063212032], [-0.009638202693937002, 0.009638202693936993], [-0.00869915884422953, 0.008699158844229537], [-0.006112192536081592, 0.006112192536081593], [-0.0022481502236240277, 0.002248150223624022], [-0.001638873920073574, 0.0016388739200735723], [-0.0012930954913152259, 0.0012930954913152241], [3.237635146592907e-05, -3.2376351465932993e-05], [0.0029760515912889396, -0.0029760515912889422]])
		model.bias = np.array([[2.39114614,-2.39114614]])
		mask_img = model.test(img)

		#selem = disk(8)
		#mask_img = closing(mask_img,selem=selem)

		return np.uint8(mask_img)

	def get_bounding_box(self, img):
		'''
			Find the bounding box of the blue barrel
			call other functions in this class if needed
			
			Inputs:
				img - original image
			Outputs:
				boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
				where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively. The order of bounding boxes in the list
				is from left to right in the image.
				
			Our solution uses xy-coordinate instead of rc-coordinate. More information: http://scikit-image.org/docs/dev/user_guide/numpy_images.html#coordinate-conventions
		'''
		boxes = []
		binary_img = self.segment_image(img)
		#clean up image
		#find connected components
		contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)		
        #iterate through all the top-level contours,
		#draw each connected component with its own random color
		for idx in range(len(contours)):
			color = 255*np.random.random([3])
			cv2.drawContours(binary_img, contours, idx, color, -1 )
		
		#go through each region
		#find apply shape statistic to include or exclude as barrel
		props = regionprops(binary_img)
		for reg in props:
			print(reg.area)
			#make sure area seen is sizable enough
			if reg.area > 400:
				major = reg.major_axis_length
				minor = reg.minor_axis_length + 0.001
				ratio = major/minor
				print(ratio)
				#make sure area is shaped like barrel (longer than wider)
				if ratio <= 2.5 and ratio >= 1.5:
					y1, x1, y2, x2 = reg.bbox
					boxes.append([x1,y1,x2,y2])

		return boxes


if __name__ == '__main__':
	folder = "trainset"
	my_detector = BarrelDetector()
	for filename in os.listdir(folder):
		# read one test image
		img = cv2.imread(os.path.join(folder,filename))
		#cv2.imshow('image', img)
		#cv2.waitKey(0)
		#cv2.destroyAllWindows()

		#Display results:
		#(1) Segmented images
		mask_img = my_detector.segment_image(img)
		#(2) Barrel bounding box		
		boxes = my_detector.get_bounding_box(img)
		#The autograder checks your answers to the functions segment_image() and get_bounding_box()
		#Make sure your code runs as expected on the testset before submitting to Gradescope
		

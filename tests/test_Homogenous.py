import moyra as ma

p = ma.DynamicModelParameters(2)
p.L_1 = ma.ModelSymbol(value = 1, string = 'L_1') # the length from the origin to the first mass
p.L_2 = ma.ModelSymbol(value = 1, string = 'L_2') # the ength from the 1st to the 2nd mass
m_1_frame = ma.HomogenousTransform().R_x(p.q[0]).Translate(0,0,-p.L_1)
m_2_frame = m_1_frame.R_x(p.q[1]).Translate(0,0,-p.L_2)
quit()
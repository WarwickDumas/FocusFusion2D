

	// What's this? : [[still in file]]
	NiTi = area* THIRD*m_ion*n_ionrec.n_ionise*((v_ion_k-v_n_k).dot(v_ion_k-v_n_k));
	NnTn = area* THIRD*m_ion*n_ionrec.n_recombine*((v_ion_k-v_n_k).dot(v_ion_k-v_n_k));
	NeTe = area* THIRD*m_e*(n_ionrec.n_ionise + n_ionrec.n_recombine)*((v_e_k-v_n_k).dot(v_e_k-v_n_k));
	// Think 1/3 must be right, and changed model document - it did not have 1/3 for NeTe.

	{
		// RESISTIVE:
		
	// Resistive htg:
	// ---------------	
	// Simpson: do 1/6 of x-fer at v_k, 2/3 after 1/2 acceleration, 1/6 after all acceleration.
	
	// our xfer: vn += -h(me/(me+mn))*nu_ne_MT*(vn-ve)
	//           ve += -h(mn/(me+mn))*nu_en_MT*(ve-vn)
	//      Te change: +h(me*mn/(me+mn))*nu_en_MT*(vn-ve).(vn-ve)
	//                                   ^^ includes n_n
	
		Tens1.xx = h*nu_eiBar ;
		Tens1.yy = Tens1.xx;
		Tens1.zz = Tens1.xx;
		f64 fac = -h*0.9*nu_eiBar*nu_eiBar/(nu_eHeart*total);
		Tens1.xx += fac*(omega_ce.x*omega_ce.x + nu_eHeart*nu_eHeart);
		Tens1.yy += fac*(omega_ce.y*omega_ce.y + nu_eHeart*nu_eHeart);
		Tens1.zz += fac*(omega_ce.z*omega_ce.z + nu_eHeart*nu_eHeart);			
		Tens1.xy = fac*(omega_ce.x*omega_ce.y - nu_eHeart*omega_ce.z);
		Tens1.xz = fac*(omega_ce.x*omega_ce.z + nu_eHeart*omega_ce.y);
		Tens1.yx = fac*(omega_ce.x*omega_ce.y + nu_eHeart*omega_ce.z);
		Tens1.yz = fac*(omega_ce.y*omega_ce.z - nu_eHeart*omega_ce.x);
		Tens1.zx = fac*(omega_ce.x*omega_ce.z - nu_eHeart*omega_ce.y);
		Tens1.zy = fac*(omega_ce.y*omega_ce.z + nu_eHeart*omega_ce.x);
						
		// This was e-i resistive heating:
		NeTe += 
			area* SIXTH*n_e_plus*TWOTHIRDS*m_e*(
			// rate of change of ve. dot(ve-vi), integrated:
						(Tens1*(v_e_k-v_ion_k)).dot(v_e_k-v_ion_k)
					+ 
						(Tens1*(v_e_k-v_ion_k+v_e_plus-v_ion_plus)).dot
							(v_e_k-v_ion_k+v_e_plus-v_ion_plus) // 0.25 cancels with 4
					+   (Tens1*(v_e_plus-v_ion_plus)).dot(v_e_plus-v_ion_plus)
					);
			
		// where's i-n?
	}
	{
		// Inelastic frictional heating:
		
		// Maybe this is actually FRICTIONAL heating e-n, i-n ;
		// I think that's what we're actually looking at here.

		f64 M_in = m_n*m_ion/((m_n+m_ion)*(m_n+m_ion));
		f64 M_en = m_n*m_e/((m_n+m_e)*(m_n+m_e));
		f64 M_ie = m_ion*m_e/((m_ion+m_e)*(m_ion+m_e));
		
		NeTe += area * SIXTH*n_e_plus*TWOTHIRDS*m_e*(
			h*(m_n/(m_e+m_n))*nu_ne_MT_over_n*nT_neut_use.n*(
				  (v_e_k-v_n_k).dot(v_e_k-v_n_k)
				+ (v_e_k-v_n_k + v_e_plus - v_n_plus).dot(v_e_k-v_n_k + v_e_plus - v_n_plus)
				+ (v_e_plus-v_n_plus).dot(v_e_plus-v_n_plus)
											));
		
		f64 v_ni_diff_sq = SIXTH*((v_n_k-v_ion_k).dot(v_n_k-v_ion_k)
				+ (v_n_k-v_ion_k+v_n_plus-v_ion_plus).dot(v_n_k-v_ion_k+v_n_plus-v_ion_plus)
				+ (v_n_plus-v_ion_plus).dot(v_n_plus-v_ion_plus));
		
		NiTi += area * n_ion_plus*TWOTHIRDS*m_n*
						h*M_in*nu_ni_MT_over_n*nT_neut_use.n*v_ni_diff_sq;
		
		NnTn += area * n_n_plus*TWOTHIRDS*m_ion*
						h*M_in*nu_ni_MT_over_n*nT_ion_use.n*v_ni_diff_sq;
				
		// We can deduce T_k+1 afterwards from n_k+1 T_k+1.
		// OR, we can rearrange conservative equations to be for T_k+1.
	}

	// We then say, in our heating routine:

	nnTn += neut_resistive/area;
	niTi += ion_resistive/area;
	neTe += elec_resistive/area;

	// Think this through. nT is per area, heat density.

	// So we need an area element.


	// .....................................................................................


	// Note that there is a way to save on an access, if we put h* visccond heatrate into
	// the same slots as central cells' resistive heating.
	// That is sensible.


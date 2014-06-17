function dummy = mlinprog(pyinfo)
	nb_input = pyinfo.nb_input;
	dir = '/tmp/mats/';
	dummy = 0;
	parfor idx=1:nb_input
		filename = sprintf('%s%s_%d.mat', dir, 'lpin', idx-1);
		if (exist(filename, 'file') == 2)
			args = load(filename);
			A=args.A;
			b=args.b;
			% A = [];
			% b = [];
			f=args.f;
			% Aeq=args.Aeq;
			% beq=args.beq;
			Aeq = [];
			beq = [];
			lb=zeros(1, numel(f));
			ub = [];
			% x0=args.x0;
			x0 = [];
			tic;
			% opts = optimoptions(@linprog,'Display','final','Algorithm','simplex');
			x = linprog(f, A, b, Aeq, beq, lb, ub);
			toc
			dst = f*x;
			oSaveVars(sprintf('%s%s_%d', dir, 'lpout', idx-1), dst);
		end
	end
end

function oSaveVars(fname, dst)
	save(fname, 'dst')
end

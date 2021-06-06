/*
	Fractal by HTML5 UP
	html5up.net | @ajlkn
	Free for personal and commercial use under the CCA 3.0 license (html5up.net/license)
*/

(function($) {

	const $window = $(window),
		$body = $('body');

	// Breakpoints.
		breakpoints({
			xlarge:   [ '1281px',  '1680px' ],
			large:    [ '981px',   '1280px' ],
			medium:   [ '737px',   '980px'  ],
			small:    [ '481px',   '736px'  ],
			xsmall:   [ '361px',   '480px'  ],
			xxsmall:  [ null,      '360px'  ]
		});

	// Play initial animations on page load.
		$window.on('load', function() {
			window.setTimeout(function() {
				$body.removeClass('is-preload');
			}, 100);
		});

	// Mobile?
		if (browser.mobile)
			$body.addClass('is-mobile');
		else {

			breakpoints.on('>medium', function() {
				$body.removeClass('is-mobile');
			});

			breakpoints.on('<=medium', function() {
				$body.addClass('is-mobile');
			});

		}

	// Scrolly.
		$('.scrolly')
			.scrolly({
				speed: 1500
			});

	// Exclude product IDs
		function exceptionrange(element) {
			element.onchange = () => {
				const excluded_products = [1012, 1021, 1045, 1047, 1055, 1059, 1073, 1087, 1089, 1102, 1104, 1110,
					1120, 1159, 1174, 1181, 1186, 1207, 1210, 1216, 1218, 1221, 1224, 1233, 1243, 1250, 1257, 1263,
					1281, 1307, 1311]
				console.log(typeof excluded_products[0])
				console.log(typeof element.value)
				console.log(excluded_products.includes(element.value))

				if (excluded_products.includes(parseInt(element.value))) {
					alert('Product with ID: ' + element.value + ' is not available. \n' +
						'It had to be removed from the database due to faulty data.')
				}
			};
		}
		
		window.addEventListener("load",() => {
			const input = document.getElementById("product_input");
			exceptionrange(input);
		});

})(jQuery);